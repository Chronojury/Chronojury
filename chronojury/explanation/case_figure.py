"""Render a standalone model input/output figure from measured case diagnostics."""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_figure(study, output_stem, font_path):
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    with plt.rc_context({'font.family': family, 'axes.unicode_minus': False, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(20, 14), facecolor='#f5f7fb')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        blue, green, red, muted = '#285d9a', '#16816b', '#c44240', '#566579'

        def text(x, y, content, size=14, color='#243246', **kwargs):
            return ax.text(x, y, content, fontsize=size, color=color,
                           transform=ax.transAxes, va='top', **kwargs)

        def box(x, y, w, h, title, body, color=blue):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.008',
                                       facecolor='white', edgecolor=color, linewidth=1.4))
            text(x + .012, y + h - .014, title, 17, color, fontweight='bold')
            text(x + .012, y + h - .053, body, 14, linespacing=1.55)

        def arrow(start, end):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=17,
                                         color='#8090a5', linewidth=1.7))

        def wrap(value, width=28):
            return '\n'.join(textwrap.wrap(value.replace('\n', ' '), width=width))

        original = study['probes']['original']
        predicted = '买家' if original['predicted_class'] == 0 else '卖家'
        target = '买家' if study['reference_label'] == 0 else '卖家'
        stages = study['stages']
        agents = len(stages[0]['opinions'])
        correct = sum(item['stance'] == study['reference_label'] for stage in stages for item in stage['opinions'])
        total = sum(len(stage['opinions']) for stage in stages)
        text(.025, .974, f"错误案例 {study['case_id']}  |  从证据输入到最终裁决", 25, fontweight='bold')
        text(.025, .933, f"平台参考：支持{target}   ·   {correct}/{total} 条阶段意见与参考一致   ·   最终模型：支持{predicted}",
             17, muted)
        claims = study['claims']
        media = study['evidence']
        body = (wrap('买家：' + claims['buyer']) + '\n' + wrap('卖家：' + claims['seller'])
                + f"\n证据条目 {len(media)} 条（含商品背景）"
                + f"\n原始媒体：{sum(e['images'] for e in media)} 张图片、{sum(e['videos'] for e in media)} 段视频")
        buyer_evidence = next((item['text'] for item in media if item['identity'] == '买家'), '')
        if '说明：' in buyer_evidence:
            excerpt = buyer_evidence.split('说明：', 1)[1].splitlines()[0]
            body += '\n' + wrap('买家凭证摘录：' + (excerpt[:26] + '…' if len(excerpt) > 26 else excerpt))
        box(.025, .69, .29, .205, '01  原始案件输入', body)
        entries = []
        for stage, boundary in zip(stages, study['boundaries']):
            identities = list(dict.fromkeys(media[index]['identity'] for index in boundary))
            entries.append(f"阶段 {stage['stage']}：证据 {boundary}（{'、'.join(identities)}）")
            if stage['perceptions']:
                entries.append(f"检索命中 {stage['retrieval_hits']} 次，感知 {len(stage['perceptions'])} 次")
            else:
                entries.append('纯文本直接进入裁决，未触发多模态感知')
        box(.355, .69, .29, .205, '02  SWER：分段、检索与感知', '\n'.join(entries))
        entries = []
        for stage in stages:
            votes = [opinion['stance'] for opinion in stage['opinions']]
            entries.append(f"阶段 {stage['stage']}：买家 {votes.count(0)} / 卖家 {votes.count(1)} / 未解析 {votes.count(None)}")
        entries += ['', '原始意见与历史日志逐条核对。', '最终分类不使用多数投票规则。']
        box(.685, .69, .29, .205, '03  智能体原始裁决', '\n'.join(entries), green)
        arrow((.317, .792), (.348, .792)); arrow((.647, .792), (.678, .792))
        ax.plot([.83, .83, .17], [.68, .665, .665], color='#8090a5', linewidth=1.7)
        arrow((.17, .665), (.17, .642))

        shapes = study['tensor_shapes']
        def shape(key):
            return ' × '.join(map(str, shapes[key]))
        entries = [f"{agents} 个意见节点 + 1 个阶段案件节点", f"编码特征：[{shape('input_features')}]", '']
        for stage in stages:
            focus = stage['case_summary'].split('争议焦点:')[-1].strip()
            if len(focus) > 42:
                focus = focus[:41] + '…'
            entries.append(wrap(f"阶段 {stage['stage']} 摘要焦点：{focus}"))
        box(.025, .405, .29, .235, '04  阶段摘要与节点编码', '\n'.join(entries))
        entries = [f"输入投影：[{shape('projected_features')}]",
                   f"空间 GAT：[{shape('gat_outputs')}]",
                   f"每阶段 {shapes['edge_index'][1]} 条有向边 + 内部自环",
                   '方向：被关注者 → 关注者，案件 → 智能体',
                   f"因果时序模块：[{shape('temporal_outputs')}]",
                   f"末阶段节点均值：[{shape('pooled_features')}]",
                   '完整中间数值另存 stage_tensors.npz']
        box(.355, .405, .29, .235, '05  STVA：空间与时序聚合', '\n'.join(entries))
        votes = study.get('platform_votes')
        entries = ["分类 logits（买家 / 卖家）：",
                   f"{original['logits'][0]:+.4f} / {original['logits'][1]:+.4f}",
                   f"P(买家) = {1-original['seller_probability']:.2%}",
                   f"P(卖家) = {original['seller_probability']:.2%}",
                   f"输出：支持{predicted}，与参考标签不一致",
                   f"平台投票（买家 / 卖家）：{votes}" if votes else f'参考标签：支持{target}']
        box(.685, .405, .29, .235, '06  原始模型输出：错误', '\n'.join(entries), red)
        arrow((.317, .522), (.348, .522)); arrow((.647, .522), (.678, .522))

        prefix = ' → '.join(f"{p['seller_probability']:.1%}" for p in study['prefix_probes'])
        body = (f"已观察：{correct}/{total} 条意见与平台参考一致，\n最终学习分类却与参考相反。\n\n"
                '定位范围：从意见编码到 STVA 最终分类的链路。\n'
                '旁路时序与屏蔽跨阶段注意力是不同操作。\n'
                '单案例无法证明某一层是唯一原因。\n\n'
                f'阶段前缀探针 P(卖家)：\n{prefix}')
        box(.025, .095, .425, .25, '定位结论与证据边界', body, blue)
        box(.48, .095, .495, .25, '冻结权重对照：卖家 softmax 分数', '', blue)
        names = ['original', 'without_case_messages', 'without_juror_messages',
                 'without_cross_stage_attention', 'readout_jurors_only', 'bypass_temporal_same_head']
        labels = ['原始完整模型', '屏蔽案件消息', '屏蔽智能体消息', '屏蔽跨阶段注意力',
                  '池化排除案件节点', '绕过整个时序模块*']
        bars = fig.add_axes([.68, .123, .255, .17])
        values = [100 * study['probes'][name]['seller_probability'] for name in names]
        colors = [red if study['probes'][name]['predicted_class'] != study['reference_label'] else green for name in names]
        bars.barh(range(len(names)), values, color=colors, height=.62)
        bars.set(yticks=range(len(names)), yticklabels=labels, xlim=(0, 108))
        bars.invert_yaxis(); bars.axvline(50, color='#9ba7b7', linestyle='--', linewidth=1)
        for i, value in enumerate(values):
            bars.text(value + 1, i, f'{value:.2f}%', va='center', fontsize=12, color='#243246')
        bars.set_xticks([0, 50, 100], ['0%', '50%', '100%']); bars.tick_params(axis='both', labelsize=12, length=0)
        for spine in bars.spines.values():
            spine.set_visible(False)
        text(.025, .057, '* 诊断性推理对照，未重新训练。旁路会改变分类头输入分布，不能据此认定时序模块存在代码错误。', 13, muted)
        text(.025, .032, '阶段前缀探针使用同一终局分类头。示例仅说明一种失败模式，不代表整体错误分布。图中分数未经概率校准。', 13, muted)
        output_stem = Path(output_stem)
        fig.savefig(output_stem.with_suffix('.png'), dpi=180)
        fig.savefig(output_stem.with_suffix('.pdf'))
        plt.close(fig)
