"""
批量辩论生成器 - 为所有案例生成辩论结果JSON文件
用途: 将辩论过程与模型训练解耦,避免重复API调用
"""
import json
import os
from collections import deque
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from . import config
from .debate import Debate
from ..tracing import trace_case
from ..runtime import MAX_CASE_WORKERS, case_lock, runtime_options
from tqdm import tqdm

class BatchDebateGenerator:
    """批量辩论生成器"""
    
    def __init__(self, 
                 output_dir: str = config.BATCH_CONFIG["output_dir"],
                 model_name: str = config.BATCH_CONFIG["model"],
                 T: float = config.BATCH_CONFIG["temperature"],
                 sleep: float = config.BATCH_CONFIG["sleep"],
                 max_workers: int = config.BATCH_CONFIG["max_workers"],
                 skip_existing: bool = True,
                 max_completion_tokens: int | None = None,
                 prefetch_cases: int = config.BATCH_CONFIG["prefetch_cases"]):
        """
        初始化批量辩论生成器
        
        Args:
            output_dir: 输出目录 (默认 Results，与 debate.py 保持一致)
            model_name: 模型名称
            T: 温度参数
            sleep: API调用间隔
            max_workers: 最大并行工作线程数
            skip_existing: 是否跳过已存在的结果文件
            max_completion_tokens: 每次裁决的最大生成 token 数
            prefetch_cases: CPU 提前读取的额外案例数，0 表示关闭预取
        """
        if prefetch_cases < 0:
            raise ValueError("prefetch_cases must be nonnegative")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.T = T
        self.sleep = sleep
        self.max_workers = max_workers
        self.prefetch_cases = prefetch_cases
        self.skip_existing = skip_existing
        self.max_completion_tokens = (config.BATCH_CONFIG["max_completion_tokens"]
                                      if max_completion_tokens is None else max_completion_tokens)
        
        self.total_cases = 0
        self.processed_cases = 0
        self.success_cases = 0
        self.failed_cases = []
        self.skipped_cases = []
        
    def _load_case_data(self, case_dir: Path) -> Dict[str, Any]:
        """加载单个案例数据"""
        data_file = case_dir / "data.json"
        
        with open(data_file, 'r', encoding='utf-8') as f:
            case_data = json.load(f)
        
        case_data['_file_path'] = str(data_file)
        case_data['_case_id'] = case_dir.name
        
        return case_data
    
    def _existing_case_file(self, case_dir: Path):
        if self.skip_existing:
            return next(self.output_dir.glob(f"{case_dir.name}_data_*_for_gnn.json"), None)
        return None

    def _prefetch_case_data(self, case_dir: Path):
        # Skipped cases may have missing or invalid inputs; do not open them.
        if self._existing_case_file(case_dir) is not None:
            return None
        return self._load_case_data(case_dir)

    @trace_case("reason", case_arg=1)
    def _process_single_case(self, case_dir: Path, case_data=None) -> Dict[str, Any]:
        with case_lock(self.output_dir, case_dir.name):
            return self._process_case(case_dir, case_data)

    def _process_case(self, case_dir: Path, case_data=None) -> Dict[str, Any]:
        """处理单个案例的辩论"""
        
        case_id = case_dir.name
        
        existing_file = self._existing_case_file(case_dir)
        if existing_file is not None:
            print(f"⏭️  跳过案例 {case_id}: 结果文件已存在 ({existing_file.name})")
            self.skipped_cases.append(case_id)
            return {
                'case_id': case_id,
                'status': 'skipped',
                'reason': 'file_exists',
                'existing_file': str(existing_file)
            }
        
        if case_data is None:
            case_data = self._load_case_data(case_dir)
        
        assert case_data.get('whowins') is not None, f"案例 {case_id}: 缺少标签"

        
        
        debate = Debate(
            model_name=self.model_name,
            T=self.T,
            sleep=self.sleep,
            case_data=case_data,
            output_dir=self.output_dir,
            max_completion_tokens=self.max_completion_tokens,
        )
        
        print(f"🎭 开始处理案例 {case_id}...")
        start_time = time.time()
        
        try:
            result = debate.run(
                case_data=case_data,
                case_path=case_dir / "data.json"
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ 案例 {case_id}: 处理异常，未保存结果 ({elapsed_time:.1f}s)")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            self.failed_cases.append({
                'case_id': case_id,
                'reason': 'exception',
                'error': str(e)
            })
            return {
                'case_id': case_id,
                'status': 'failed',
                'reason': 'exception',
                'error': str(e),
                'elapsed_time': elapsed_time
            }
        
        elapsed_time = time.time() - start_time
        
        if result is None :
            print(f"❌ 案例 {case_id}: 辩论失败")
            self.failed_cases.append({
                'case_id': case_id,
                'reason': 'debate_failed'
            })
            return {
                'case_id': case_id,
                'status': 'failed',
                'reason': 'debate_failed'
            }
        
        
        
        saved_path = result.get('saved_path') if isinstance(result, dict) else None
        print(f"✅ 案例 {case_id}: 成功 (耗时 {elapsed_time:.1f}s)")
        print(f"   💾 文件已保存到 {saved_path or self.output_dir}")
        self.success_cases += 1
        
        return {
            'case_id': case_id,
            'status': 'success',
            'note': 'saved_by_engine_video',
            'saved_path': saved_path,
            'elapsed_time': elapsed_time
        }
            
    
    def process_directory(self, data_dir: str, start_idx: int = None, end_idx: int = None) -> Dict[str, Any]:
        """
        处理整个目录的案例
        
        Args:
            data_dir: 数据目录路径
        
        Returns:
            处理结果统计
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        case_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
        case_dirs.sort(key=lambda x: int(x.name))

        if start_idx is not None or end_idx is not None:
            py_start = (start_idx - 1) if (start_idx is not None and start_idx > 0) else 0
            py_end = end_idx if end_idx is not None else len(case_dirs)
            
            sliced_case_dirs = case_dirs[py_start:py_end]
            
            print(f"✂️ [范围切片激活] 原始总案例数: {len(case_dirs)} 个")
            print(f"🎯 已成功锁定自定义目标区间: 第 {start_idx or 1} 个 到 第 {end_idx or len(case_dirs)} 个")
            print(f"📦 本次实际需要炼丹的案例文件夹为: {[d.name for d in sliced_case_dirs]}")
            
            case_dirs = sliced_case_dirs
        
        self.total_cases = len(case_dirs)
        if self.total_cases == 0:
            print("⚠️ 警告：当前切片区间内没有找到任何合法的案例文件夹！")
            raise ValueError("No cases matched the requested input range")
        print(f"\n📊 找到 {self.total_cases} 个案例")
        print(f"💾 输出目录: {self.output_dir}")
        print(f"⚡ 并行线程数: {self.max_workers}")
        print(f"📥 CPU 预取案例数: {self.prefetch_cases}")
        print(f"⏭️  跳过已存在: {self.skip_existing}\n")
        
        start_time = time.time()
        
        results = []
        live = bool(os.environ.get('CHRONOJURY_RUNTIME_CONFIG'))
        reader_pool = ThreadPoolExecutor(max_workers=1) if self.prefetch_cases or live else nullcontext()
        worker_limit = MAX_CASE_WORKERS if live else self.max_workers
        with reader_pool as reader, ThreadPoolExecutor(max_workers=worker_limit) as executor:
            def process_prefetched(case_dir, loaded):
                return self._process_single_case(case_dir, loaded.result())

            case_iter = iter(case_dirs)
            prepared = deque()
            pending = set()
            exhausted = False
            previous_options = None
            with tqdm(total=len(case_dirs), desc="处理案例") as pbar:
                while pending or prepared or not exhausted:
                    options = runtime_options()
                    workers = min(worker_limit, options.get('reason_workers', self.max_workers))
                    prefetch = options.get('prefetch_cases', self.prefetch_cases)
                    current_options = (workers, prefetch)
                    if current_options != previous_options:
                        print(f'Runtime scheduling: {workers} cases, {prefetch} CPU-prefetched inputs', flush=True)
                        previous_options = current_options
                    # Prefetch inputs without starting extra GPU/API work.
                    needed = max(0, workers - len(pending)) + prefetch
                    while not exhausted and len(prepared) < needed:
                        case_dir = next(case_iter, None)
                        if case_dir is None:
                            exhausted = True
                            break
                        loaded = reader.submit(self._prefetch_case_data, case_dir) if reader is not None else None
                        prepared.append((case_dir, loaded))
                    while prepared and len(pending) < workers:
                        case_dir, loaded = prepared.popleft()
                        future = (executor.submit(process_prefetched, case_dir, loaded) if loaded is not None
                                  else executor.submit(self._process_single_case, case_dir))
                        pending.add(future)
                    if not pending:
                        continue
                    finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in finished:
                        results.append(future.result())
                        self.processed_cases += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            '成功': self.success_cases,
                            '失败': len(self.failed_cases),
                            '跳过': len(self.skipped_cases)
                        })

        total_time = time.time() - start_time
        
        report = {
            'summary': {
                'total_cases': self.total_cases,
                'processed_cases': self.processed_cases,
                'success_cases': self.success_cases,
                'failed_cases': len(self.failed_cases),
                'skipped_cases': len(self.skipped_cases),
                'total_time': total_time,
                'avg_time_per_case': total_time / max(self.processed_cases, 1)
            },
            'failed_cases': self.failed_cases,
            'skipped_cases': self.skipped_cases,
            'all_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        report_dir = Path(os.environ.get('CHRONOJURY_REPORT_DIR', self.output_dir))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / 'batch_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("🎉 批量辩论生成完成!")
        print("=" * 60)
        print(f"📊 总案例数: {self.total_cases}")
        print(f"✅ 成功: {self.success_cases}")
        print(f"❌ 失败: {len(self.failed_cases)}")
        print(f"⏭️  跳过: {len(self.skipped_cases)}")
        print(f"⏱️  总耗时: {total_time:.1f}s")
        print(f"⚡ 平均耗时: {total_time / max(self.processed_cases, 1):.1f}s/案例")
        print(f"📝 报告已保存: {report_file}")
        print("=" * 60)
        
        return report
