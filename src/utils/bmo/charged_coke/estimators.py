# ruff: noqa
# flake8: noqa
"""Training estimators for the charged-coke model (vendored, unmodified).

Copied byte-for-byte from the BF-02 research package
(``BF02_Charged_Coke_Model_Package.zip`` -> ``charged_final/estimators.py``,
SHA-256 7c1d57bfe48a0d353cd15dfc5d38dd57536ae683b3c825c67196c5a9d251c538) so models retrained here are fitted exactly as the study fitted
them. Only the code below the marker is the original; do not reformat it.

* ``SequenceTransform``: median imputation, standardisation, clip to +/-6.
* ``LinearDelta``: ridge on sequence summaries (multi-output).
* ``DeltaAttention``: NumPy attention with one query per output horizon.

Inference in the app does not use these classes: fitted weights are exported
to arrays and evaluated by ``utils.bmo.charged_coke.model``.
"""
# ---- original research code below ------------------------------------------
import numpy as np
from scipy.special import softmax,expit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge,HuberRegressor,ElasticNet
from sklearn.multioutput import MultiOutputRegressor

class SequenceTransform:
    def fit(self,x):
        f=x.shape[-1];a=x.reshape(-1,f)
        self.med=np.nanmedian(a,axis=0);self.med=np.nan_to_num(self.med)
        a=np.where(np.isfinite(a),a,self.med)
        self.mean=a.mean(0);self.std=a.std(0);self.std[self.std<1e-5]=1
        return self
    def transform(self,x):return np.clip((np.where(np.isfinite(x),x,self.med)-self.mean)/self.std,-6,6)

def summarize(x):
    """Every summary uses only the input history; channel blocks stay ordered."""
    return np.concatenate([x[:,-1],x[:,-1]-x[:,-2],x[:,-1]-x[:,-5],x[:,-1]-x[:,0],np.mean(x[:,-4:],axis=1),np.mean(x,axis=1),np.std(x[:,-4:],axis=1)],axis=1)

class LinearDelta:
    def __init__(self,kind='ridge',alpha=100):self.kind=kind;self.alpha=alpha
    def fit(self,x,y,weights=None,**kwargs):
        self.trans=SequenceTransform().fit(x);a=summarize(self.trans.transform(x))
        self.scale=StandardScaler().fit(a);a=self.scale.transform(a)
        if self.kind=='ridge':self.model=Ridge(alpha=self.alpha)
        elif self.kind=='huber':self.model=MultiOutputRegressor(HuberRegressor(alpha=self.alpha,epsilon=1.35,max_iter=300))
        elif self.kind=='elastic':self.model=MultiOutputRegressor(ElasticNet(alpha=self.alpha,l1_ratio=.25,max_iter=2000))
        self.model.fit(a,y,sample_weight=weights)
        return self
    def predict(self,x):return self.model.predict(self.scale.transform(summarize(self.trans.transform(x))))

class DeltaAttention:
    """State-conditioned feature gates, relative encoding and horizon queries.

    Optional auxiliary direction task and change-weighted loss are explicit
    variants. Attention weights are diagnostics, never causal attribution.
    """
    def __init__(self,dim=12,seed=17,epochs=100,lr=.003,decay=.005,relative=True,skip=True,loss='huber',aux=0.,change_weight=1.,selection='mae'):
        self.dim=dim;self.seed=seed;self.epochs=epochs;self.lr=lr;self.decay=decay;self.relative=relative;self.skip=skip;self.loss=loss;self.aux=aux;self.change_weight=change_weight;self.selection=selection
    def _init(self,f,l,o):
        rng=np.random.default_rng(self.seed);d=self.dim
        self.p={'Wg':rng.normal(0,.025,(f,f)),'bg':np.zeros(f),'Wa':rng.normal(0,.10,(f,d)),'Wr':rng.normal(0,.06,(f,d)),'be':np.zeros(d),
        'Wq':rng.normal(0,.13,(o,d,d)),'bt':np.zeros((o,l)),'wo':rng.normal(0,.015,(o,2*d)),'bo':np.zeros(o),
        'Ws':np.zeros((3*f,o)),'wc':rng.normal(0,.02,(o,2*d,3)),'bc':np.zeros((o,3))}
    def _forward(self,x,cache=False):
        p=self.p;g=expit(x[:,-1]@p['Wg']+p['bg']);z=x*g[:,None,:];rel=z-z[:,-1:]
        h=np.tanh(z@p['Wa']+(rel@p['Wr'] if self.relative else 0)+p['be'])
        q=np.einsum('nd,odk->nok',h[:,-1],p['Wq'])
        score=np.einsum('nld,nod->nol',h,q)/np.sqrt(self.dim)+p['bt']
        a=softmax(score,axis=2);ctx=np.einsum('nol,nld->nod',a,h)
        u=np.concatenate([ctx,np.repeat(h[:,-1,None,:],len(p['bo']),axis=1)],axis=2)
        summary=np.concatenate([x[:,-1],x[:,-1]-x[:,-5],x[:,-1]-x.mean(axis=1)],axis=1)
        pred=np.einsum('nod,od->no',u,p['wo'])+p['bo']+(summary@p['Ws'] if self.skip else 0)
        prob=softmax(np.einsum('nod,odc->noc',u,p['wc'])+p['bc'],axis=2)
        return (pred,prob,(x,g,z,rel,h,q,a,u,summary)) if cache else pred
    def _grad(self,x,y,weights,classes):
        pred,prob,(x,g,z,rel,h,q,a,u,summary)=self._forward(x,True);p=self.p;d=self.dim;o=y.shape[1]
        w=weights[:,None]*(1+(self.change_weight-1)*(classes!=1));den=max(w.sum(),1e-9)
        err=pred-y;dp=(np.clip(err,-1.5,1.5) if self.loss=='huber' else err)*w/den
        gr={k:np.zeros_like(v) for k,v in p.items()}
        gr['wo']=np.einsum('no,nod->od',dp,u);gr['bo']=dp.sum(0)
        if self.skip:gr['Ws']=summary.T@dp
        du=dp[:,:,None]*p['wo'][None]
        if self.aux:
            dc=prob.copy();ni,oi=np.indices(classes.shape);dc[ni,oi,classes]-=1
            cw=self.class_weights[oi,classes];dc*=self.aux*weights[:,None,None]*cw[:,:,None]/(weights.sum()*o)
            gr['wc']=np.einsum('nod,noc->odc',u,dc);gr['bc']=dc.sum(0);du+=np.einsum('noc,odc->nod',dc,p['wc'])
        dc=du[:,:,:d];dh=np.einsum('nol,nod->nld',a,dc);dh[:,-1]+=du[:,:,d:].sum(axis=1)
        da=np.einsum('nod,nld->nol',dc,h);ds=a*(da-(da*a).sum(axis=2,keepdims=True))
        gr['bt']=ds.sum(0);dh+=np.einsum('nol,nod->nld',ds,q)/np.sqrt(d)
        dq=np.einsum('nol,nld->nod',ds,h)/np.sqrt(d)
        gr['Wq']=np.einsum('nd,nok->odk',h[:,-1],dq);dh[:,-1]+=np.einsum('nok,odk->nd',dq,p['Wq'])
        de=dh*(1-h*h);gr['Wa']=np.einsum('nlf,nld->fd',z,de);gr['be']=de.sum((0,1));dz=de@p['Wa'].T
        if self.relative:
            gr['Wr']=np.einsum('nlf,nld->fd',rel,de);dr=de@p['Wr'].T;dz+=dr;dz[:,-1]-=dr.sum(1)
        dg=(dz*x).sum(1)*g*(1-g);gr['Wg']=x[:,-1].T@dg;gr['bg']=dg.sum(0)
        for k in gr:
            if k in ['Wg','Wa','Wr','Wq','wo','Ws','wc']:gr[k]+=self.decay*p[k]
        return gr
    def _train(self,x,y,w,classes,epochs,val=None):
        rng=np.random.default_rng(self.seed);m={k:np.zeros_like(v) for k,v in self.p.items()};v={k:np.zeros_like(v) for k,v in self.p.items()};step=0;best=np.inf;bestep=1;stale=0
        self.class_weights=np.ones((y.shape[1],3))
        for j in range(y.shape[1]):
            counts=np.bincount(classes[:,j],minlength=3);self.class_weights[j]=np.clip(np.sqrt(len(y)/(3*np.maximum(counts,1))),.5,4)
        for ep in range(1,epochs+1):
            perm=rng.permutation(len(x))
            for k in range(0,len(x),128):
                ix=perm[k:k+128];grad=self._grad(x[ix],y[ix],w[ix],classes[ix]);step+=1;norm=np.sqrt(sum(np.square(a).sum() for a in grad.values()));clip=min(1,5/max(norm,1e-9))
                for key in self.p:
                    a=grad[key]*clip;m[key]=.9*m[key]+.1*a;v[key]=.999*v[key]+.001*a*a
                    self.p[key]-=self.lr*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
            if val is not None:
                e=self._forward(val[0])-val[1];vv=val[2][:,None]
                loss=np.sum(vv*(np.abs(e) if self.selection=='mae' else e*e))/(vv.sum()*e.shape[1])
                if loss<best-1e-4:best=loss;bestep=ep;stale=0
                else:stale+=1
                if ep>=25 and stale>=14:break
        return bestep if val is not None else epochs
    def fit(self,x,y,weights=None,classes=None):
        weights=np.ones(len(x)) if weights is None else weights
        classes=np.ones_like(y,dtype=int) if classes is None else classes
        cut=max(40,int(len(x)*.8));end=max(20,cut-12)
        self.trans=SequenceTransform().fit(x[:end]);xx=self.trans.transform(x)
        self._init(x.shape[-1],x.shape[1],y.shape[1]);ep=self._train(xx[:end],y[:end],weights[:end],classes[:end],self.epochs,(xx[cut:],y[cut:],weights[cut:]))
        self.trans=SequenceTransform().fit(x);xx=self.trans.transform(x);self._init(x.shape[-1],x.shape[1],y.shape[1]);self._train(xx,y,weights,classes,ep);self.epochs_=ep
        return self
    def predict(self,x):return self._forward(self.trans.transform(x))
    def explain(self,x):
        pred,prob,cache=self._forward(self.trans.transform(x),True)
        return {'prediction':pred,'direction_probabilities':prob,'feature_gates':cache[1],'time_attention':cache[6]}
