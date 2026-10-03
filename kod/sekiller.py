"""Sunumdaki grafikleri üretir.  Kullanım: python3 kod/sekiller.py data/CWRU-dataset
(Önce kod/cwru_deney.py çalıştırılmış ve sonuclar/sonuclar.json oluşmuş olmalı.)"""
import sys, json, os
import numpy as np, scipy.signal as ss, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
exec(open(os.path.join(os.path.dirname(__file__),'cwru_deney.py')).read().split('# ---------------------------------------------------------------- model')[0])
B,O,A,Y,INK,MUT='#2a78d6','#eb6834','#1baf7a','#eda100','#1f2328','#6b7280'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.edgecolor':'#bbb','axes.labelcolor':INK,'xtick.color':'#555','ytick.color':'#555'})
def clean(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(axis='y',color='#eee'); ax.set_axisbelow(True)
D='sekiller/'; os.makedirs(D, exist_ok=True); tr=lambda v:('%.1f'%v).replace('.',',')
fr=1748/60; BPFO=3.5848*fr
xn=recording('N',None,2); xo=recording('OR','007',2)
# 1 time signals
fig,axs=plt.subplots(2,1,figsize=(11,5),dpi=160,sharex=True,sharey=True)
t=np.arange(4800)/FS*1000
for ax,x,c,lab in [(axs[0],xn,MUT,'Sağlam rulman'),(axs[1],xo,O,'Dış bilezik hasarı (0,007")')]:
    ax.plot(t,x[:4800],color=c,lw=0.8); clean(ax); ax.grid(False)
    ax.text(0.005,1.02,f'{lab}   ·   kurtosis {tr(kurtosis(x,fisher=False))}',transform=ax.transAxes,fontsize=12,color=INK,va='bottom',fontweight='bold')
    ax.set_ylabel('İvme (g)')
axs[1].set_ylim(-5.6,5.6); axs[1].set_xlabel('Zaman (ms)')
axs[1].text(0.995,0.06,'darbeler arası ≈ 9,6 ms = 1 / BPFO',transform=axs[1].transAxes,ha='right',fontsize=11,color=INK,bbox=dict(fc='white',ec='none',pad=2))
plt.tight_layout(); plt.savefig(D+'1_zaman_sinyali.png',bbox_inches='tight'); plt.close()
# 2 envelope spectrum
x=xo[:240000]; b,a=ss.butter(4,[2000,4500],btype='band',fs=FS)
e=np.abs(ss.hilbert(ss.filtfilt(b,a,x))); e-=e.mean()
E=2*np.abs(np.fft.rfft(e*np.hanning(len(e))))/len(e)*2; f=np.fft.rfftfreq(len(e),1/FS)
Sr=2*np.abs(np.fft.rfft(x*np.hanning(len(x))))/len(x)*2
fig,axs=plt.subplots(1,2,figsize=(12,4.3),dpi=160)
m=f<24000; axs[0].plot(f[m]/1000,Sr[m],color=B,lw=0.6); axs[0].set_title('Ham spektrum: enerji kHz bölgesinde',loc='left',fontsize=12,fontweight='bold')
axs[0].set_xlabel('Frekans (kHz)'); axs[0].set_ylabel('Genlik'); axs[0].axvspan(2,4.5,color=Y,alpha=0.25); axs[0].text(6.5,axs[0].get_ylim()[1]*0.9,'← filtre bandı\n(rezonans)',ha='left',fontsize=9,color=INK)
m=f<450; axs[1].plot(f[m],E[m],color=A,lw=1)
for k,lab in [(1,'BPFO'),(2,'2×'),(3,'3×'),(4,'4×')]:
    axs[1].axvline(k*BPFO,color=O,ls='--',lw=1); axs[1].text(k*BPFO+3,axs[1].get_ylim()[1]*0.92,lab,color=O,fontsize=10)
axs[1].set_title('Zarf spektrumu: tepe 105,0 Hz (hesap: 104,4)',loc='left',fontsize=12,fontweight='bold'); axs[1].set_xlabel('Frekans (Hz)')
for ax in axs: clean(ax)
plt.tight_layout(); plt.savefig(D+'2_zarf_spektrumu.png',bbox_inches='tight'); plt.close()

# 3 fingerprints FB_20 mean, size 007
fig,axs=plt.subplots(1,4,figsize=(13,3.6),dpi=160,sharey=True)
for ax,(k,s,lab,c) in zip(axs,[('N',None,'Sağlam',MUT),('IR','007','İç bilezik',O),('OR','007','Dış bilezik',B),('B','007','Bilye',A)]):
    W=windows(recording(k,s,2)); F=fs_FB20_mean(W,spectrum(W)).mean(0)
    ax.bar(np.arange(20),F,color=c,width=0.8); ax.set_yscale('log'); ax.set_title(lab,fontsize=13,fontweight='bold',loc='left')
    ax.set_xticks([0,5,10,15,19],['0','6','12','18','24']); ax.set_xlabel('kHz'); clean(ax)
axs[0].set_ylabel('Bant ortalaması (log)')
plt.tight_layout(); plt.savefig(D+'3_bant_parmak_izi.png',bbox_inches='tight'); plt.close()
r=json.load(open('sonuclar/sonuclar.json'))
# 4 confusion matrix
cm=np.array(r['D2_cm']['FB_20_FD']); P=100*cm/cm.sum(1,keepdims=True); L=['Sağlam','İç bilezik','Dış bilezik','Bilye']
fig,ax=plt.subplots(figsize=(6.6,5.4),dpi=160)
ax.imshow(P,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b',['#ffffff','#cde2fb','#6da7ec','#2a78d6','#184f95']),vmin=0,vmax=100)
for i in range(4):
    for j in range(4): ax.text(j,i,'%'+str(int(round(P[i,j]))),ha='center',va='center',fontsize=14,color='white' if P[i,j]>55 else INK,fontweight='bold' if i==j else 'normal')
ax.set_xticks(range(4),L); ax.set_yticks(range(4),L); ax.set_xlabel('Modelin tahmini'); ax.set_ylabel('Gerçek durum')
ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
for s in ax.spines.values(): s.set_visible(False)
ax.add_patch(Rectangle((-0.5,-0.5),1,1,fill=False,ec=A,lw=3))
plt.tight_layout(); plt.savefig(D+'4_karisiklik_matrisi.png',bbox_inches='tight'); plt.close()
# 5 load generalization
rpm=r['rpm']; xs=[0,1,2,3]; xl=[f"{L} hp\n{int(rpm[str(L)])} rpm" for L in xs]
fig,ax=plt.subplots(figsize=(9,4.8),dpi=160)
for n,c,lab in [('FB_20_FD',B,'FB_20_FD'),('FB_20_FD-mean',A,'FB_20_FD-mean'),('RAW_Lei',MUT,'RAW_Lei'),('FB_20_rel (öneri)',O,'FB_20_rel (öneri)')]:
    v=[100*r['D5'][n]['0hp'],100*r['D5'][n]['1hp'],100*r['D1'][n][0],100*r['D5'][n]['3hp']]
    ax.plot(xs,v,color=c,lw=2,marker='o',ms=8,label=lab)
ax.axvspan(1.6,2.4,color=Y,alpha=0.18); ax.text(2,62,'eğitim\nyükü',ha='center',fontsize=10,color=INK)
ax.set_xticks(xs,xl); ax.set_ylim(60,102); ax.set_ylabel('Doğruluk (%)'); clean(ax)
ax.legend(frameon=False,fontsize=10,loc='lower right')
plt.tight_layout(); plt.savefig(D+'5_yuk_genellemesi.png',bbox_inches='tight'); plt.close()
# 6 leakage diagram
fig,axs=plt.subplots(1,2,figsize=(12,3.6),dpi=160)
rng=np.random.default_rng(3)
for ax,title,mode in [(axs[0],'Rastgele bölme (makale)','rand'),(axs[1],'Kayıt bazlı bölme (sızıntısız)','rec')]:
    for i,s in enumerate(['0,007"','0,014"','0,021"']):
        y=2-i; ax.text(-0.5,y+0.4,s,ha='right',va='center',fontsize=11,color=INK)
        for w in range(40):
            te=(rng.random()<0.2) if mode=='rand' else (i==1)
            ax.add_patch(Rectangle((w*0.25,y+0.1),0.22,0.6,color=O if te else B))
    ax.set_xlim(-2.2,10.2); ax.set_ylim(-0.2,3.2); ax.axis('off'); ax.set_title(title,fontsize=13,fontweight='bold',loc='left')
fig.text(0.5,0.02,'■ eğitim     ■ test      (her kutu = 34,8 ms\'lik bir pencere, her satır = bir rulman kaydı)',ha='center',fontsize=10,color=MUT)
fig.texts[-1].set_text('')
fig.text(0.33,0.03,'■',color=B,fontsize=14); fig.text(0.345,0.035,'eğitim',fontsize=11)
fig.text(0.42,0.03,'■',color=O,fontsize=14); fig.text(0.435,0.035,'test',fontsize=11)
fig.text(0.5,0.035,'· her kutu = 34,8 ms\'lik bir pencere · her satır = bir rulman kaydı',fontsize=10,color=MUT)
plt.tight_layout(rect=(0,0.08,1,1)); plt.savefig(D+'6_sizinti_semasi.png',bbox_inches='tight'); plt.close()
# 7 ana sonuç grafiği
names=list(r['D1']); lab=['RAW_Lei\n(25 istatistik)','FB_20_FD\n(280)','FB_20_FD-mean\n(20)','FB_20_rel\n(öneri, 20)']
d1=[100*r['D1'][n][0] for n in names]; d2=[100*r['D2'][n]['mean'] for n in names]
fig,ax=plt.subplots(figsize=(10,5.6),dpi=160); x=np.arange(4); w=0.36
b1=ax.bar(x-w/2-0.01,d1,w,color=B,label='Makalenin protokolü (rastgele 5 katlı CV, 10 sınıf)')
b2=ax.bar(x+w/2+0.01,d2,w,color=O,label='Sızıntısız (görülmemiş hata boyutu, 4 sınıf)')
for bs in (b1,b2):
    for b in bs: ax.text(b.get_x()+b.get_width()/2,b.get_height()+1.2,'%'+tr(b.get_height()),ha='center',fontsize=10,color='#333')
ax.axhline(25,color='#888',lw=1,ls='--'); ax.text(-0.45,21.5,'4 sınıfta şans seviyesi: %25',ha='left',fontsize=9,color='#555',bbox=dict(fc='white',ec='none',pad=1))
ax.set_xticks(x,lab); ax.set_ylim(0,108); ax.set_xlim(-0.5,3.5); ax.set_ylabel('Doğruluk (%)'); clean(ax)
ax.set_title('Aynı özellikler, aynı model, farklı değerlendirme: %99,9 → %64',loc='left',fontsize=13,fontweight='bold')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=1,frameon=False,fontsize=10)
plt.tight_layout(); plt.savefig(D+'7_ana_sonuc.png',bbox_inches='tight'); plt.close()
print('sekiller/ klasörüne 7 grafik yazıldı')
