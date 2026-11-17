"""Figures use the same fixed geometric cutaway and sample for every panel."""
from pathlib import Path
import json,colorsys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from segmentation_pipeline import CLASS_NAMES,PALETTE
ROOT=Path(__file__).parent

def make_figures(root=ROOT):
    with np.load(root/'data/processed/Area_3_office_1.npz') as d:
        xyz=d['xyz'].copy();rgb=d['rgb'].copy();truth=d['semantic'].copy()
    with np.load(root/'results/pretrained_area3/segmentation.npz') as d:
        pred=d['semantic'].copy();groups=d['group_id'].copy()
    lo=xyz.min(0);hi=xyz.max(0)
    # Same coordinate-only cutaway for RGB, truth, prediction and grouping.
    visible=(xyz[:,2]<lo[2]+.86*(hi[2]-lo[2]))&(xyz[:,1]>lo[1]+.10)&(xyz[:,0]<hi[0]-.10)
    indices=np.flatnonzero(visible)
    indices=np.random.default_rng(42).choice(indices,min(40000,len(indices)),replace=False)
    point=xyz[indices]-lo
    group_color=np.array([colorsys.hsv_to_rgb((int(i)*.61803398875)%1,.65,.92) if i>=0 else (.55,.55,.55) for i in groups[indices]])
    error_color=np.where((pred[indices]!=truth[indices])[:,None],np.array([[.90,.14,.18]]),np.array([[.78,.80,.83]]))
    group_count=len(set(groups)-{-1})
    panels=[('Real scan RGB',rgb[indices]/255.),('Ground truth',PALETTE[truth[indices]]/255.),
            ('RandLA-Net prediction',PALETTE[pred[indices]]/255.),(f'{group_count} geometric candidate groups',group_color),
            ('Classification errors (red)',error_color)]
    fig=plt.figure(figsize=(16,10),facecolor='white')
    for i,(title,color) in enumerate(panels):
        ax=fig.add_subplot(2,3,i+1,projection='3d')
        ax.scatter(*point.T,c=color,s=.55,depthshade=False,rasterized=True)
        ax.set_box_aspect(hi-lo);ax.view_init(27,-62)
        ax.set_title(title,fontsize=12,pad=10)
        ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
        ax.tick_params(labelsize=7)
    ax=fig.add_subplot(2,3,6);ax.axis('off')
    ax.legend(handles=[Patch(color=PALETTE[i]/255.,label=n) for i,n in enumerate(CLASS_NAMES)],loc='upper left',ncol=2,frameon=False,fontsize=11)
    ax.text(0,.40,'Area 3 / office 1 | 759,861 source points\n\nIdentical cutaway and 40,000-point sample.\nCeiling/front boundaries hidden only for viewing.\nAll source points are retained in the output.\n\nGroups are candidates for human review;\ntouching objects may merge and surfaces may split.',va='top',fontsize=11,linespacing=1.6)
    fig.suptitle('Indoor point cloud: semantics and editable grouping candidates',fontsize=18,y=.99)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(root/'results/segmentation_overview.png',dpi=150);plt.close(fig)
    rf=json.loads((root/'results/random_forest/report.json').read_text())
    net=json.loads((root/'results/pretrained_area3/report.json').read_text())
    fig,ax=plt.subplots(figsize=(13,4.5));x=np.arange(13)
    for shift,report,label,color in [(-.19,rf,'RF: trained on one room','#94a3b8'),(.19,net,'Pretrained RandLA-Net','#2874a6')]:
        values=[report['evaluation']['class_iou'][n] for n in CLASS_NAMES]
        ax.bar(x+shift,[0 if v is None else v*100 for v in values],width=.38,label=label,color=color)
    ax.set_xticks(x,CLASS_NAMES,rotation=25);ax.set_ylim(0,105);ax.set_ylabel('IoU (%)')
    ax.set_title('Same test room; training data volume differs (not an architecture ablation)')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.02,.01,'Column: absent in truth but false predictions exist (IoU = 0). Sofa: absent in truth and both predictions (N/A).',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(root/'results/class_iou_comparison.png',dpi=150);plt.close(fig)
    return [root/'results/segmentation_overview.png',root/'results/class_iou_comparison.png']

if __name__=='__main__':
    for path in make_figures():print(path)
