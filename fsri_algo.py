import pandas as pd
import numpy as np

class PQR:
    def __init__(self,n_components,quantile):
        self.factor_num = n_components
        self.quantile = quantile

    def fit(self,X,Y):
        T,N= X.shape
        self.features = N
        y = np.array(Y).reshape(-1,1).copy()
        factor = self.factor_num

        # standardize X
        self.stdizer = StandardScaler().fit(X)
        X_std = self.stdizer.transform(X)
        self.wts = np.full([N,factor],np.nan)
        # First-pass quantile regressions
        # Preallocate Phi
        Factor = np.full([T,factor],np.nan) # T * factor_num
        Phi  = np.full([N,1],np.nan) # factor_num *1
        self.sec_pass_params_x = np.full([factor,N],np.nan)
        iota = np.ones([T,1])
        x = X_std.copy()
        for f in np.arange(factor):
            for i in np.arange(N):
                X_uni = x[:,i]
                X_uni = sm.add_constant(X_uni)
                model_1 = sm.QuantReg(y, X_uni).fit(q=self.quantile)
                Phi[i] = model_1.params[1]
            self.wts[:,f] = Phi[:,0] 
            # Second-pass OLS regressions
            # Preallocate F
            F = np.full([T,1],np.nan)
            Phi = sm.add_constant(Phi)
            for t in np.arange(T):
                model_2 = sm.OLS(x[t,:], Phi).fit()
                F[t] = model_2.params[1]
            # X,Y 都去掉前次factor的效果～之後再去建構下一個factor
            y = sm.OLS(y,F).fit().resid
            model_x_resid = sm.OLS(x,F).fit()
            # save second-pass's params for later new data transform to F_new
            self.sec_pass_params_x[f] = model_x_resid.params
            x = model_x_resid.resid
            Factor[:,f] = F[:,0]
        self.pqr_factor = Factor
        # Third-pass quantile regression
        F_constant = sm.add_constant(Factor)
        self.m_pqr_3pass = sm.QuantReg(Y, F_constant).fit(q=self.quantile)
        fit_is = np.dot(F_constant,self.m_pqr_3pass.params)
        fit_is = np.insert(fit_is, 0, np.nan, axis=0)
        return fit_is.reshape(-1,1)

    def predict(self, X_new):
        # input : array, X_new (N,1)
        # output: array, F_new (factor_num,1)
        # check & stdize data
        X_new_std = (X_new-self.stdizer.mean_)/self.stdizer.scale_
        X_new_std = X_new_std.reshape(self.features,)
        factors = self.factor_num
        
        # initializing new factor
        F_new = np.full([self.factor_num, 1],np.nan) #size = num of factor*1
        wts = sm.add_constant(self.wts) # fit_ins_sec-pass's result
        for f in np.arange(factors):
            model_new = sm.OLS(X_new_std, wts[:,[0,f+1]]).fit()  #fitted in the first pass size = (3,1)
            F_new[f] = model_new.params[1]
            # remove the effect of previous process
            X_new_std -= (self.sec_pass_params_x[f]*F_new[f])

        self.F_new = F_new
        return np.dot(np.insert(self.F_new,0,1),self.m_pqr_3pass.params) #

    def transform(self,X_new):
        self.predict(X_new)
        return self.F_new

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests

# X: 

# x_name, y_name must be list
class dimension_reduction:
    def __init__(self, data,y_name, oos_start=None, innovation=False ,quantile=0.5, algorithm=None):

        self.algo = algorithm
        self.df = data
        self.index_series = data.index
        self.endgo = np.array(self.df[y_name]).reshape(-1,len(y_name))
        self.period_num = self.endgo.shape[0]
        self.quantile = quantile
        self.uncond_series = np.full([self.period_num,1],np.nan)
        self.full_fitted = np.full([self.period_num,1],np.nan)
        self.oos_fitted = np.full([self.period_num,1],np.nan)

        if innovation==True:
            self.__innovation()
        else:
            pass

        if oos_start==None:
            self.oos_start = self.period_num
        else:
            self.oos_start = oos_start
        
        self.get_unconditional_y()

    def __innovation(self):
        # input: Y array
        # output: aic, best_order of AR, innovation array
        y = self.endgo
        # Iterate over all ARMA(p,q) to get best p,q
        result = sm.tsa.arma_order_select_ic(y,6,0,'aic') # maxp=6, maxq=0, select by aic
        p,q = result['aic_min_order']
        model_arma = sm.tsa.ARIMA(y, order=(p, 0, q)).fit()
        # arma's resid as endgo's groth shock
        self.endgo = np.array(model_arma.resid).reshape(-1,1)


    def mqr_fit(self,X,Y):
        self.stdizer = StandardScaler().fit(X)
        X_std = self.stdizer.transform(X)
        X_std_ = sm.add_constant(X_std)
        self.m_mqr = sm.QuantReg(Y,X_std_).fit(self.quantile)
        _ = (self.m_mqr.predict(X_std_))
        _ = np.insert(_, 0,np.nan)
        return _.reshape(-1,1)

    def mqr_predict(self, X_new):
        X_new_std = self.stdizer.transform(X_new)
        _ = np.insert(X_new_std,0,1)
        return self.m_mqr.predict(_)

    def pcaqr_fit(self, X, Y):
        self.stdizer = StandardScaler().fit(X) # pca需要先行標準化
        X_std = self.stdizer.transform(X)
        self.m_pca = PCA(n_components=self.factor_num)
        self.factor = self.m_pca.fit_transform(X_std)
        l_c = sm.add_constant(self.factor)
        self.m_pcaqr = sm.QuantReg(Y, l_c).fit(q=self.quantile)
        b = self.m_pcaqr.params
        fitted = np.dot(l_c, b).reshape(-1,1)
        fitted = np.insert(fitted, 0, np.nan,axis=0)
        return fitted

    def pcaqr_predict(self, X_new):
        X_std = self.stdizer.transform(X_new)
        l_new= self.m_pca.transform(X_std)
        l_new_c = np.insert(l_new,0,1)
        predicted = np.dot(self.m_pcaqr.params,l_new_c)
        return predicted
    def pcaqr_transform(self,X_new):
        X_std = self.stdizer.transform(X_new)
        return self.m_pca.transform(X_std)
        
    def pls_fit(self,X,Y):
        # 略過標準化因為PLS內會自行標準化
        self.m_pls = PLSRegression(n_components=self.factor_num).fit(X,Y)
        self.factor = self.m_pls.transform(X)
        fitted = np.insert(self.m_pls.predict(X), 0, np.nan, axis=0)
        return fitted

    def pls_predict(self,X_new):
        return self.m_pls.predict(X_new)
    
    def pls_transform(self,X_new):
        return self.m_pls.transform(X_new)

    def pqr_fit(self,X,Y):
        # 略過標準化PQR內自行標準化
        self.m_pqr = PQR(n_components=self.factor_num,quantile=self.quantile)
        fitted = self.m_pqr.fit(X,Y)
        self.factor = self.m_pqr.pqr_factor
        return fitted

    def pqr_predict(self, X_new):
        return self.m_pqr.predict(X_new)

    def pqr_transform(self,X_new):
        return self.m_pqr.transform(X_new)

    def __unconditional(self,y):
        zeros = np.zeros((y.shape[0],1))
        model_uncond = sm.QuantReg(y, sm.add_constant(zeros)).fit(q=self.quantile)
        uncond = model_uncond.params[0]
        return uncond 

    def get_unconditional_y(self):
        y = self.endgo
        self.uncond_series[:s] = self.__unconditional(y[:s])
        for i in np.arange(self.period_num-s):
            self.uncond_series[s+i] = self.__unconditional(y[:s+i])
        
    def oos_fit(self,x_name, factor_num):
        self.factor_num = factor_num
        s = self.oos_start
        x = np.array(self.df[x_name]).reshape(-1,len(x_name))
        y = self.endgo

        #ins
        self.oos_fitted[:s] = getattr(self,self.algo+'_fit')(x[:s], y[:s])[1:]
        #oos
        for i in np.arange(self.period_num-s):
            getattr(self,self.algo+'_fit')(x[:s+i], y[:s+i]) #fit
            self.oos_fitted[s+i] = getattr(self,self.algo+'_predict')(x[s+i].reshape(-1,x.shape[1]))
        print(f'Model: {self.algo} ||| out_of_sample recursively predict start from: {s}')
        return self.oos_fitted
    
    def sector_fit(self,sector_of_xname):
        assert len(sector_of_xname)>1, 'Should be more than 1 sector'
        # 每個部門固定只取一個factor
        self.factor_num = 1

        #self.oos_fitted_s = np.full([self.period_num,1],np.nan)
        sector_factor = np.full([self.period_num, len(sector_of_xname)],np.nan) #每個sector取一個
        s = self.oos_start
        y = self.endgo

        # get factors by sector
        for g in np.arange(len(sector_of_xname)):
            x = np.array(self.df[sector_of_xname[g]]).reshape(-1,len(sector_of_xname[g]))
            getattr(self,self.algo+'_fit')(x[:s],y[:s])
            sector_factor[:s,g] = self.factor.reshape(-1,) # getfactor
            # recursively form factor
            for i in np.arange(self.period_num-s):
                x_new = x[s+i].reshape(-1,len(sector_of_xname[g]))
                y_new = y[s+i]
                getattr(self,self.algo+'_fit')(x[:s+i], y[:s+i])
                sector_factor[s+i,g] = getattr(self,self.algo+'_transform')(x_new)
        # save sector_factor        
        self.sector_factor = sector_factor.copy()
        # predict _ins
        s_f = sm.add_constant(sector_factor)
        m_sf_ins = sm.QuantReg(y[:s],s_f[:s]).fit(q=self.quantile)
        self.oos_fitted[:s] = m_sf_ins.predict().reshape(-1,1)
        # predict _oos
        for t in np.arange(self.period_num-s):
            m_sf_oos = sm.QuantReg(y[:s+t], s_f[:s+t]).fit(q=self.quantile)
            self.oos_fitted[s+t] = m_sf_oos.predict(s_f[s+t])
        print('oos_fitted has been replaced by sector_fit')

    
    # result
    def __rsquared(self,start,end):
        # input : array, fitted value series, unconditional series
        # output: float
        q = self.quantile
        a = (self.endgo - self.oos_fitted)[start:end]
        diff_pred_mean = np.where(a>0, a*q, -a*(1-q)).mean()
        b = (self.endgo - self.uncond_series)[start:end]
        diff_uncon_mean = np.where(b>0, b*q, -b*(1-q)).mean()
        Psedo_R2 = 1 - (diff_pred_mean / diff_uncon_mean)
        return Psedo_R2
    def break_through(self):
        s = self.oos_start
        self.ins_bt = sum(self.oos_fitted[:s] > self.endgo[:s]) / len(self.endgo[:s])
        if s < self.period_num:
            self.oos_bt = sum(self.oos_fitted[s:] > self.endgo[s:]) / len(self.endgo[s:])
        else:
            self.oos_bt = 0

    # result
    def __r2(self):
        self.ins_r2 = self.__rsquared(0,self.oos_start)
        self.oos_r2 = self.__rsquared(self.oos_start,self.period_num)

    
    def GS_test(self, maxlags):
        # Note
        # second column Granger causes the time series in the first column
        # https://www.statsmodels.org/0.9.0/generated/statsmodels.tsa.stattools.grangercausalitytests.html
        grangercausalitytests(np.concatenate((self.endgo,self.oos_fitted),axis=1) ,maxlag=maxlags)

    def show(self):
        self.__r2()
        self.break_through()
        print(f'Model: {self.algo} ')
        print('---------Result---------')
        print(f'In Sample: \n > R2 : {round(self.ins_r2,4)*100} % \n > B_T%: {round(self.ins_bt[0],4)*100} %')
        if self.oos_start < self.period_num:
            print(f'Out Of Sample: \n > R2: {round(self.oos_r2,4)*100} % \n > B_T%: {round(self.oos_bt[0],4)*100} %')
        else:
            pass