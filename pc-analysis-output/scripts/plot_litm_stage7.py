"""Experiment-7 PNG/SVG figures from derived tables only."""
from pathlib import Path
import csv
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'pc-analysis-output/stages/07_litm'
for key in ['MPLCONFIGDIR','XDG_CACHE_HOME','TEMP','TMP']:
    os.environ[key]=str(OUT/('tmp' if key in ['TEMP','TMP'] else 'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['sequential','ape_fixed','ape_dynamic']
LABELS=['顺序编码','固定APE','动态APE']
COLORS=['#4c78a8','#54a24b','#f58518']
POSITIONS=['0','9','19']


def readcsv(name):
    with (OUT/'tables'/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    quality=readcsv('quality.csv');paired=readcsv('quality_paired.csv')
    def q(method,pos,metric='official_first_line_em'):
        return next(r for r in quality if r['method']==method and r['position']==pos and r['score_metric']==metric)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for method,label,color,marker in zip(METHODS,LABELS,COLORS,['o','s','^']):
        rows=[q(method,p) for p in POSITIONS]
        y=np.array([float(r['em_percent']) for r in rows])
        ax.errorbar([0,9,19],y,yerr=[y-[float(r['ci_low_percent']) for r in rows],[float(r['ci_high_percent']) for r in rows]-y],marker=marker,color=color,label=label,capsize=4,linewidth=1.8)
    ax.set_xticks([0,9,19],['位置0（第1篇）','位置9（第10篇）','位置19（第20篇）'])
    ax.set_xlim(-1,20);ax.set_ylim(40,80)
    ax.set_ylabel('官方首行EM / %');ax.set_xlabel('gold资料在20篇资料中的位置（零基）')
    ax.set_title('500个共同问题的位置曲线（95%问题配对Bootstrap区间）')
    ax.legend();ax.grid(axis='y',alpha=.2)
    save(fig,'official_position_curve')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['saved_full_answer_em','official_first_line_em'],['原保存口径：完整回答','正式主表：官方首行']):
        for method,label,color,marker in zip(METHODS,LABELS,COLORS,['o','s','^']):
            ax.plot([0,9,19],[float(q(method,p,metric)['em_percent']) for p in POSITIONS],marker=marker,label=label,color=color)
        ax.set_xticks([0,9,19]);ax.set_xlabel('资料位置（零基）');ax.set_ylim(45,82);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('EM / %');axes[0].legend()
    fig.suptitle('评分入口改变501/4500条标签，涉及147个问题（图示为均值）')
    save(fig,'scoring_scope_comparison')

    rows=[next(r for r in paired if r['reference_method']=='ape_fixed' and r['comparison_method']=='ape_dynamic' and r['score_metric']=='official_first_line_em' and r['position']==p) for p in [*POSITIONS,'ALL']]
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for j,r in enumerate(rows):
        v,lo,hi=[float(r[k]) for k in ['delta_pp','ci_low_pp','ci_high_pp']]
        ax.errorbar(v,3-j,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color='#f58518' if j<3 else 'black')
        ax.annotate(f'{v:+.2f} [{lo:.2f}, {hi:.2f}]',(hi,3-j),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
    ax.set_yticks(range(4),['总体：每题三位置均值','位置19','位置9','位置0'])
    ax.axvline(0,color='gray',linestyle='--');ax.set_xlim(-8.5,13)
    ax.set_xlabel('动态APE − 固定APE / 官方首行EM百分点')
    ax.set_title('方法差异及95%区间：500个问题共同保留三个位置')
    save(fig,'dynamic_fixed_difference')

    contrasts=readcsv('position_contrasts.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,name,title in zip(axes,['middle_minus_ends_mean','last_minus_first'],['位置9 − 两端均值','位置19 − 位置0']):
        rows=[next(r for r in contrasts if r['score_metric']=='official_first_line_em' and r['method']==m and r['contrast']==name) for m in METHODS]
        y=np.array([float(r['contrast_pp']) for r in rows])
        ax.bar(range(3),y,color=COLORS,width=.6)
        ax.errorbar(range(3),y,yerr=[y-[float(r['ci_low_pp']) for r in rows],[float(r['ci_high_pp']) for r in rows]-y],fmt='none',ecolor='black',capsize=4)
        ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(3),LABELS)
        ax.set_ylabel('同题位置差 / EM百分点');ax.set_title(title)
    fig.suptitle('位置敏感性：中间相对两端与末位相对首位分别统计')
    save(fig,'position_contrasts')

    efficiency=readcsv('efficiency_descriptive.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,metric,title,unit in zip(axes,['ttft_seconds','total_seconds','output_tokens'],['首token时间','总响应时间','输出长度（含EOS）'],['秒','秒','token']):
        rows=[next(r for r in efficiency if r['position']=='ALL' and r['method']==m and r['metric']==metric) for m in METHODS]
        y=np.array([float(r['mean']) for r in rows])
        ax.bar(range(3),y,color=COLORS)
        ax.errorbar(range(3),y,yerr=[y-[float(r['mean_ci_low']) for r in rows],[float(r['mean_ci_high']) for r in rows]-y],fmt='none',ecolor='black',capsize=4)
        ax.set_xticks(range(3),LABELS);ax.set_ylabel(unit);ax.set_title(title)
    fig.suptitle('每方法1500次冷请求均值（CI按500问题保留三位置重采样）')
    save(fig,'cold_efficiency')

    limits=readcsv('generation_limits.csv');impact=readcsv('scoring_scope_impact.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    rr=[next(r for r in limits if r['position']=='ALL' and r['method']==m) for m in METHODS]
    done=np.array([int(r['cap_after_first_newline_count']) for r in rr]);unfinished=np.array([int(r['cap_unterminated_first_line_count']) for r in rr]);eos_only=np.array([int(r['cap_hits']) for r in rr])-done-unfinished
    axes[0].bar(range(3),done,color='#4c78a8',label='已出现换行')
    axes[0].bar(range(3),unfinished,bottom=done,color='#f58518',label='首行未换行且无EOS')
    axes[0].bar(range(3),eos_only,bottom=done+unfinished,color='#54a24b',label='无换行但以EOS终止')
    axes[0].set_xticks(range(3),LABELS);axes[0].set_ylabel('命中128-token上限的请求数');axes[0].set_title('上限记录的首行完成状态');axes[0].legend(fontsize=8)
    matrix=np.array([[int(next(r['changed_generations'] for r in impact if r['method']==m and r['position']==p))/500*100 for p in POSITIONS] for m in METHODS])
    im=axes[1].imshow(matrix,cmap='Oranges',vmin=0,vmax=20,aspect='auto')
    for i in range(3):
        for j in range(3):axes[1].text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center')
    axes[1].set_xticks(range(3),POSITIONS);axes[1].set_yticks(range(3),LABELS)
    axes[1].set_xlabel('gold位置');axes[1].set_title('全文EM→首行EM改变比例（各格500次）')
    fig.colorbar(im,ax=axes[1],label='评分改变比例 / %')
    save(fig,'cap_and_scoring_effects')

    requests=readcsv('per_request_records.csv')
    by={(r['sample_id'],r['method'],r['position']):r for r in requests}
    ids=sorted({r['sample_id'] for r in requests})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,ref,title in zip(axes,['sequential','ape_fixed'],['动态相对顺序','动态相对固定']):
        points=[(by[(sid,ref,p)],by[(sid,'ape_dynamic',p)]) for sid in ids for p in POSITIONS]
        xx=[int(b['output_tokens'])-int(a['output_tokens']) for a,b in points]
        yy=[float(b['total_seconds'])-float(a['total_seconds']) for a,b in points]
        zz=[(float(b['official_first_line_em'])-float(a['official_first_line_em']))*100 for a,b in points]
        im=ax.scatter(xx,yy,c=zz,cmap='RdBu_r',vmin=-100,vmax=100,s=10,alpha=.45,edgecolors='none')
        ax.axhline(0,color='gray',linewidth=.6);ax.axvline(0,color='gray',linewidth=.6)
        ax.set_xlabel('输出token差（比较−参照）');ax.set_ylabel('总响应时间差 / 秒');ax.set_title(title)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='官方首行EM差 / 百分点')
    fig.suptitle('每点为同题同位置的一对请求；重采样仍以问题为单位')
    save(fig,'paired_output_latency')

    cases=readcsv('selected_case_positions.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,label in zip(axes,['改善','退化','持平']):
        rr=sorted((r for r in cases if r['case_type']==label),key=lambda r:int(r['position']))
        matrix=np.array([[float(r[k]) for r in rr] for k in ['fixed_official_em','dynamic_official_em']])
        im=ax.imshow(matrix,cmap='YlGn',vmin=0,vmax=1,aspect='auto')
        for i in range(2):
            for j in range(3):ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center',color='white' if matrix[i,j] else 'black')
        ax.set_xticks(range(3),POSITIONS);ax.set_yticks(range(2),['固定APE','动态APE']);ax.set_xlabel('资料位置')
        ax.set_title(f"{label} · {rr[0]['sample_id']}")
    fig.colorbar(im,ax=axes.ravel().tolist(),label='官方首行EM',ticks=[0,1])
    fig.suptitle('按每题三位置均分选例：所有位置共同保留')
    save(fig,'selected_case_positions')
    print('8 experiment-7 figures written in PNG and SVG.')


if __name__=='__main__':
    main()
