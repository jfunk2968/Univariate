# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 08:44:22 2015

@author: Jeremy
"""



import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as st   
import matplotlib.pyplot as plt
import univariate_functions as uf
import time
from random import sample
import time


"""  ---------------------------------------------------------  """
"""  Class Definitions                                          """
"""  ---------------------------------------------------------  """    

class Char_Var:
          
    def __init__(self,var,target,name):
        self.name = name
        self.counts = var.value_counts()
        self.unique = len(self.counts)
        
    def print_details(self):
        print '-------------------------'
        print 'Variable Name            :   '+str(self.name)
        print 'Number of Unique Values  :   '+str(self.unique)
        print 
        print self.counts
        print 
        print '-------------------------'
        

class Char_Var_Binary(Char_Var):
          
    def __init__(self,var,target,name):
        t=time.time()
        Char_Var.__init__(self,var,target,name)
        self.stats = uf.stats_char(var,target,target_type='binary')
        self.info_value = uf.info_value(self.stats,'Count','N_Target')
        print "     "+str(time.time()-t)+"  -  "+str(self.name)
        
    def plot(self,fig=None):
        if fig==None:
            fig=self.name
        plt.close(fig)
        fign = plt.figure(fig)
        ax1 = fign.add_subplot(111)
        ax1.bar(self.stats.index,self.stats['Count'],.5,color='b',align='center')
        ax2 = ax1.twinx()
        ax2.plot(self.stats.index,self.stats['Target Rate'],'-r',linewidth=4)
        plt.title(self.name+" - Univariate View")
        ax1.set_xlabel(' Value')
        ax1.set_xticks(self.stats.index)
        ax1.set_xticklabels(self.stats['Value'], rotation=40, ha='right')
        ax1.set_ylabel('Number of Observations',color='b')
        ax2.set_ylabel('Target Rate', color='r')
        for t1 in ax1.get_yticklabels():
            t1.set_color('b')  
        for t2 in ax2.get_yticklabels():
            t2.set_color('r')     
        plt.show(fig)


class Char_Var_Continuous(Char_Var):
          
    def __init__(self,var,target,name):
        t=time.time()
        Char_Var.__init__(self,var,target,name)
        self.stats = uf.stats_char(var,target,target_type='continuous')
        self.anova_value = uf.F_test_char(var,target)
        print "     "+str(time.time()-t)+"  -  "+str(self.name)
        
    def plot(self,fig=None):
        if fig==None:
            fig=self.name
        plt.close(fig)
        fign = plt.figure(fig)
        ax1 = fign.add_subplot(111)
        ax1.bar(self.stats.index,self.stats['Count'],.5,color='b',align='center')
        ax2 = ax1.twinx()
        ax2.plot(self.stats.index,self.stats['Mean_Target'],'-r',linewidth=4)
        plt.title(self.name+" - Univariate View")
        ax1.set_xlabel(' Value')
        ax1.set_xticks(self.stats.index)
        ax1.set_xticklabels(self.stats['Value'], rotation=40, ha='right')
        ax1.set_ylabel('Number of Observations',color='b')
        ax2.set_ylabel('Mean Target', color='r')
        for t1 in ax1.get_yticklabels():
            t1.set_color('b')  
        for t2 in ax2.get_yticklabels():
            t2.set_color('r')     
        plt.show(fig)



class Num_Var:
       
    def __init__(self,var,target,bins,name):
        self.name = name
        self.summary = var.describe()
        self.corr = pearsonr(var.astype(float),target)[0]
        
    def print_details(self):
        print '-------------------------'
        print 'Variable Name          :   '+str(self.name)
        print 'Correlation to Target  :   '+str(self.corr)
        print 
        print self.summary
        print 
        print '-------------------------'
        
    
class Num_Var_Binary(Num_Var):
       
    def __init__(self,var,target,bins,name):
        t=time.time()
        Num_Var.__init__(self,var,target,bins,name)
        self.stats = uf.num_bin_stats(var,target,bins,target_type='binary')    
        self.info_value = uf.info_value(self.stats,'Count','N_Target')
        print "     "+str(time.time()-t)+"  -  "+str(self.name)
        
    def plot(self,fig=None):
        if fig==None:
            fig=self.name
        plt.close(fig)
        fign = plt.figure(fig)
        ax1 = fign.add_subplot(111)
        ax1.bar(self.stats.index,self.stats['Count'],.5,color='b',align='center')
        ax2 = ax1.twinx()
        ax2.plot(self.stats.index,self.stats['Target_Rate'],'-r',linewidth=4)
        plt.title(self.name+" - Univariate View")
        ax1.set_xlabel(' bin')
        ax1.set_xticks(self.stats.index)
        ax1.set_xticklabels(self.stats['bin'], rotation=40, ha='right')
        ax1.set_ylabel('Number of Observations',color='b')
        ax2.set_ylabel('Target Rate', color='r')
        for t1 in ax1.get_yticklabels():
            t1.set_color('b')  
        for t2 in ax2.get_yticklabels():
            t2.set_color('r')     
        plt.show(fig)
                
    def plot_bubble(self,fig=None):
        if fig==None:
            fig=self.name
        plt.close(fig)
        bubbles = self.stats.loc[(self.stats['bin']!='MISS')&(self.stats['bin']!='ZERO')]
        bubbles['bubsize']=(bubbles['Count']/np.mean(bubbles['Count']))*200
        plt.close(fig)
        plt.figure(fig)
        plt.scatter(x=bubbles['Mean'],y=bubbles['Target_Rate'],s=bubbles['bubsize'],c='b')
        plt.hold(True)
        plt.plot(bubbles['Mean'],bubbles['Target_Rate'],'-r',linewidth=4)
        plt.title(self.name+" - Relational Bubble Plot")
        plt.xlabel('Mean')
        plt.ylabel('Target Rate',color='r')
        plt.legend(['Target Rate','Population Size'],4).draggable()
        plt.show(fig)       
        
        
        
class Num_Var_Continuous(Num_Var):
       
    def __init__(self,var,target,bins,name):
        t=time.time()
        Num_Var.__init__(self,var,target,bins,name)
        self.stats = uf.num_bin_stats(var,target,bins,target_type='continuous') 
        b = self.stats[['bin']]
        self.anova_value = uf.F_test_num(var,target,b)
        print "     "+str(time.time()-t)+"  -  "+str(self.name)
        
    def plot(self,fig=None):
        if fig==None:   
            fig = self.name
        plt.close(fig)
        fign = plt.figure(fig)
        ax1 = fign.add_subplot(111)
        ax1.bar(self.stats.index,self.stats['Count'],.5,color='b',align='center')
        ax2 = ax1.twinx()
        ax2.plot(self.stats.index,self.stats['Mean_Target'],'-r',linewidth=4)
        plt.title(self.name+" - Univariate View")
        ax1.set_xlabel(' bin')
        ax1.set_xticks(self.stats.index)
        ax1.set_xticklabels(self.stats['bin'], rotation=40, ha='right')
        ax1.set_ylabel('Number of Observations',color='b')
        ax2.set_ylabel('Mean_Target', color='r')
        for t1 in ax1.get_yticklabels():
            t1.set_color('b')  
        for t2 in ax2.get_yticklabels():
            t2.set_color('r')     
        plt.show(fig)
                
    def plot_bubble(self,fig=None):
        if fig==None:   
            fig = self.name
        plt.close(fig)
        bubbles = self.stats.loc[(self.stats['bin']!='MISS')&(self.stats['bin']!='ZERO')]
        bubbles['bubsize']=bubbles['Count'].apply(lambda x: int((x/np.mean(bubbles['Count']))*200))
        plt.figure(fig)
        plt.scatter(x=bubbles['Mean'],y=bubbles['Mean_Target'],s=bubbles['bubsize'],c='b')
        plt.hold(True)
        plt.plot(bubbles['Mean'],bubbles['Mean_Target'],'-r',linewidth=4)
        plt.title(self.name+" - Relational Bubble Plot")
        plt.xlabel('Mean')
        plt.ylabel('Mean_Target',color='r')
        plt.legend(['Mean_Target','Population Size'],4).draggable()
        plt.show(fig)               
        


class Univariate:
    
    def __init__(self,name,df,target,exclude=[],numeric_bins=10):
        self.name = name
        self.df = df
        self.nrows = len(df)
        self.target = target
        self.exclude = exclude        
        self.num_vars = {}
        self.char_vars = {}  
        
        #get list of numeric variables to include in summary
        self.exclude.append(target)
        self.variables = [x for x in df.columns if x not in self.exclude]
        
        self.numeric_bins = numeric_bins
        self.n,self.c,self.u = uf.parse_dataframe(df,self.variables)  


    def print_summary(self):
        print '----------------------------------------------------------------'
        print 'Report Name              : '+str(self.name)
        print 'Number of Rows in Data   : '+str(self.nrows)
        print 'Number of Cols in Data   : '+str(len(self.df.columns))  
        print ' - Numeric (reviewed)    : '+str(len(self.num_vars))
        print ' - Character (reviewed)  : '+str(len(self.char_vars))
        print 'Target Variable          : '+str(self.target)
        print
        print 'Excluded Variables       : '+str(self.exclude)
        print        
        print 'Unary Variables          : '+str(self.u)
        print
        print 'Reviewed Variables       : '+str(self.variables)
        print '----------------------------------------------------------------'
        
            
    def create_report(self,report_name):
        None
        
        
    def _bivar_numnum(self,num1,num2,bv_bins=10,fign=None,npoints=100):
        if fign==None:   
            fign = str(num1)+" X "+str(num2)
        
        #get temp df to work with and sample for scatterplot
        temp = self.df[[num1,num2]]    
        rindex = np.array(sample(xrange(len(temp)),npoints))
        temp_samp = temp.ix[rindex]
       
        #plot scatterplot
        plt.close(fign)
        plt.figure(fign)
        plt.scatter(temp_samp[num1],temp_samp[num2])
        #plt.legend(['Target = 1','Target = 0'])
        plt.xlabel('Num1 :'+str(num1))
        plt.ylabel('Num2: '+str(num2))
        plt.title('Bivariate: Numeric by Numeric')
        
        #get stats for binned mean and bubble plot
        d = uf.num_bin_stats(self.df[num1],self.df[num2],bv_bins,target_type='continuous')
        
        #plot binned mean
        plt.close(fign+': Binned Mean')
        fig_bm = plt.figure(fign+': Binned Mean')
        ax1 = fig_bm.add_subplot(111)
        ax1.bar(d.index,d['Count'],.5,color='b',align='center')
        ax2 = ax1.twinx()
        ax2.plot(d.index,d['Mean_Target'],'-r',linewidth=4)
        plt.title(fign+': Binned Mean')
        ax1.set_xlabel(num1+' - Bin')
        ax1.set_xticks(d.index)
        ax1.set_xticklabels(d['bin'], rotation=40, ha='right')
        ax1.set_ylabel('Number of Observations',color='b')
        ax2.set_ylabel('Mean - '+num2, color='r')
        for t1 in ax1.get_yticklabels():
            t1.set_color('b')  
        for t2 in ax2.get_yticklabels():
            t2.set_color('r')     
        plt.show(fign+': Binned Mean')  
        
        #plot bubble plot
        plt.close(fign+': Bubble')
        bubbles = d.loc[(d['bin']!='MISS')&(d['bin']!='ZERO')]
        bubbles['bubsize']=bubbles['Count'].apply(lambda x: int((x/np.mean(bubbles['Count']))*200))
        plt.figure(fign+': Bubble')
        plt.scatter(x=bubbles['Mean'],y=bubbles['Mean_Target'],s=bubbles['bubsize'],c='b')
        plt.hold(True)
        plt.plot(bubbles['Mean'],bubbles['Mean_Target'],'-r',linewidth=4)
        plt.title(fign+": Bubble Plot")
        plt.xlabel(num1+' - Mean')
        plt.ylabel('Mean - '+num2,color='r')
        plt.legend(['Mean - '+num2,'Pop Size'],4).draggable()
        plt.show(fign+': Bubble')       
      
        return d
        

    def _bivar_catcat(self,cat1,cat2,fign=None):
        if fign==None:   
            fign = str(cat1)+" X "+str(cat2)
            
        #get freq table    
        ct = pd.crosstab(self.df[cat1],self.df[cat2])
        labels = ct.columns
        
        #create percentage columns
        ct['total']=ct.sum(axis=1)
        for i in range(0,len(labels)):
            ct[labels[i]+'_pct']=ct[labels[i]]/ct['total']
            

        if ((len(ct)*len(labels))>100):
            print "WARNING - Too many levels to create plots"
            
        else:    
        
            #set colors for plots
            colors=['b','r','y','g','aqua','orangered','magenta','midnightblue','lawngreen','crimson']          
            
            #plot frequencies        
            bar_width = 1/float(len(labels)+1)
            plt.figure(fign+' - Counts')
            ind = np.arange(len(ct))
            for i in range(0,len(labels)):
                plt.bar(ind+(i*bar_width),ct[labels[i]],bar_width,color=colors[i%len(colors)],label=labels[i])
            plt.title(fign+" -  Freq Dist")
            plt.xlabel(cat1+' Value')
            plt.xticks(ind+.5,ct.index,rotation=40,ha='right')
            plt.ylabel(cat2+' Count')
            plt.legend().draggable()
            plt.show(fign+' - Counts')       
            
            #plot percentages        
            bar_width = 1/float(len(labels)+1)
            plt.figure(fign+' - Percentages')
            ind = np.arange(len(ct))
            for i in range(0,len(labels)):
                plt.bar(ind+(i*bar_width),ct[labels[i]+'_pct'],bar_width,color=colors[i%len(colors)],label=labels[i])
            plt.title(fign+" -  Percent Dist")
            plt.xlabel(cat1+' Value')
            plt.xticks(ind+.5,ct.index,rotation=40,ha='right')
            plt.ylabel(cat2+' Percent')
            plt.legend().draggable()
            plt.show(fign+' - Percentages')    
        
        return ct
        
    def _bivar_numcat(self,num,cat,bins=10,fign=None):
        if fign==None:   
            fign = str(num)+" X "+str(cat)
        
        #make temp dataset with bin labels
        temp = self.df[[num,cat]]                      
        b = uf.num_bucket_assign(temp,num,bins)
        cuts = []
        for i in range(0,len(b)):
            cuts.append(b['bin'].iloc[i][0])
        cuts.append(b['bin'].iloc[len(b)-1][1])
        temp['_cats_'] = pd.cut(temp[num],bins=cuts,labels=b['bin'])
              
        #get freq table    
        ct = pd.crosstab(temp['_cats_'],temp[cat])
        labels = ct.columns            

        if ((len(ct)*len(labels))>100):
            print "WARNING - Too many levels to create plots"
            
        else:    
        
            #set colors for plots
            colors=['b','r','y','g','aqua','orangered','magenta','midnightblue','lawngreen','crimson']          
            
            #plot frequencies        
            bar_width = 1/float(len(labels)+1)
            plt.figure(fign+' - Counts')
            ind = np.arange(len(ct))
            for i in range(0,len(labels)):
                plt.bar(ind+(i*bar_width),ct[labels[i]],bar_width,color=colors[i%len(colors)],label=labels[i])
            plt.title(fign+' - Binned Dist')
            plt.xlabel(num+' Bin')
            plt.xticks(ind+.5,ct.index,rotation=40,ha='right')
            plt.ylabel(cat+' Count')
            plt.legend().draggable()
            plt.show(fign+' - Counts')       
        
        return ct
        
        
    def _bivar_catnum(self,cat,num,bins=10,fign=None):
        if fign==None:   
            fign = str(num)+" X "+str(cat)
        
        #get binned mean stats
        means = uf.stats_char(self.df[cat],self.df[num],target_type='continuous')
        
        plt.close(fign)
        fig_cn = plt.figure(fign)
        ax1 = fig_cn.add_subplot(111)
        ax1.bar(means.index,means['Count'],.5,color='b',align='center')
        ax2 = ax1.twinx()
        ax2.plot(means.index,means['Mean_Target'],'-r',linewidth=4)
        plt.title(fign+' - Mean by Cat')
        ax1.set_xlabel(cat+' Value')
        ax1.set_xticks(means.index)
        ax1.set_xticklabels(means['Value'], rotation=40, ha='right')
        ax1.set_ylabel('Counts',color='b')
        ax2.set_ylabel(num+' Mean', color='r')
        for t1 in ax1.get_yticklabels():
            t1.set_color('b')  
        for t2 in ax2.get_yticklabels():
            t2.set_color('r')     
        plt.show(fign)
        
        return means
        
    def bivar(self,var1,var2):
        if ((var1 in self.char_vars) == (var1 in self.num_vars)):
            raise TypeError(var1+" is not in a single (num or char) variable dict")
        if ((var2 in self.char_vars) == (var2 in self.num_vars)):
            raise TypeError(var2+" is not in a single (num or char) variable dict")
        if ((var1 in self.num_vars) and (var2 in self.num_vars)):
            return self._bivar_numnum(var1,var2)
        if ((var1 in self.num_vars) and (var2 in self.char_vars)):
            return self._bivar_numcat(var1,var2)
        if ((var1 in self.char_vars) and (var2 in self.num_vars)):
            return self._bivar_catnum(var1,var2)
        if ((var1 in self.char_vars) and (var2 in self.char_vars)):
            return self._bivar_catcat(var1,var2)


            
        
class Univariate_Binary(Univariate):
    
    def __init__(self,name,df,target,exclude=[],numeric_bins=10,**kwargs):
        '''   
        kwargs can include num_vars list of Num_Var objects and char_vars list of
        Char_Var objects created before hand, named "nvarin" and "cvarin" respectively
        '''
        print '----------------------------------------------------------------'        
        print "Initializing Univariate: "+str(name)
        Univariate.__init__(self,name,df,target,exclude,numeric_bins)
        
        #populate num_vars with Num_Var objects 
        if kwargs.get('nvarin') == None:
            print "Calculate Stats for Numeric Variables:"
            i=0
            for nvar in self.n:
                print "   "+str(i)+" in "+str(len(self.n))
                self.num_vars[nvar] = Num_Var_Binary(df[nvar],df[target],self.numeric_bins,nvar)
                i+=1
        else:
            self.num_vars = kwargs.get('nvarin')
            
        #create character variable summary stats table
        if kwargs.get('cvarin') == None:
            print "Calculate Stats for Character Variables:"
            i=0
            for cvar in self.c:
                print "   "+str(i)+" in "+str(len(self.c))
                self.char_vars[cvar] = Char_Var_Binary(df[cvar],df[target],cvar)
                i+=1
        else:
            self.char_vars = kwargs.get('cvarin')
        
        print "Done Initializing "
        print '----------------------------------------------------------------'

    def IV_report(self):
        IV = pd.DataFrame(index=(self.n+self.c),columns=['IV','Var Type'])     
        
        for n in self.num_vars:
            IV['IV'].ix[n] = self.num_vars[n].info_value
            IV['Var Type'].ix[n] = 'numeric'
        
        for c in self.char_vars:
            IV['IV'].ix[c] = self.char_vars[c].info_value
            IV['Var Type'].ix[c] = 'character'
        
        IV.sort(columns='IV',ascending=False,inplace=True,axis=0)
        
        return IV
        
    def print_IV_report(self):
        IV = self.IV_report()
        
        print '----------------------------------------------------------------'
        print 'IV Report for Analysis: '+str(self.name)
        print
        print IV
        print '----------------------------------------------------------------'
        
        return None

    def create_plots(self,max_vars=-1):
        a = self.IV_report()        
        
        count = 0
        for v in a.index:
            if a['Var Type'].ix[v]=='numeric':
                self.num_vars[v].plot(v)
                self.num_vars[v].plot_bubble(str(v+' - Bubble'))            
            else:
                self.char_vars[v].plot(v)  
                
            count += 1
            if (max_vars!=-1 and count>=max_vars):
                return

class Univariate_Continuous(Univariate):
    
    def __init__(self,name,df,target,exclude=[],numeric_bins=10,**kwargs):
        '''   
        kwargs can include num_vars list of Num_Var objects and char_vars list of
        Char_Var objects created before hand, named "nvarin" and "cvarin" respectively
        '''
        print '----------------------------------------------------------------'        
        print "Initializing Univariate: "+str(name)
        Univariate.__init__(self,name,df,target,exclude)
        
        #populate num_vars with Num_Var objects 
        if kwargs.get('nvarin') == None:
            print "Calculate Stats for Numeric Variables:"
            i=0
            for nvar in self.n:
                print "   "+str(i)+" in "+str(len(self.n))
                self.num_vars[nvar] = Num_Var_Continuous(df[nvar],df[target],numeric_bins,nvar)
                i+=1
        else:
            self.num_vars = kwargs.get('nvarin')
            
        #create character variable summary stats table
        if kwargs.get('cvarin') == None:
            print "Calculate Stats for Character Variables:"
            i=0
            for cvar in self.c:
                print "   "+str(i)+" in "+str(len(self.c))
                self.char_vars[cvar] = Char_Var_Continuous(df[cvar],df[target],cvar)
                i+=1
        else:
            self.char_vars = kwargs.get('cvarin')
        
        print "Done Initializing "
        print '----------------------------------------------------------------'


    def anova_report(self):
        anova = pd.DataFrame(index=(self.n+self.c),columns=['F-Test','P-Value','Var Type'])     
        
        for n in self.num_vars:
            anova['F-Test'].ix[n] = self.num_vars[n].anova_value[0]
            anova['P-Value'].ix[n] = self.num_vars[n].anova_value[1]
            anova['Var Type'].ix[n] = 'numeric'
        
        for c in self.char_vars:
            anova['F-Test'].ix[c] = self.char_vars[c].anova_value[0]
            anova['P-Value'].ix[c] = self.char_vars[c].anova_value[1]
            anova['Var Type'].ix[c] = 'character'
        
        anova.sort(columns='P-Value',inplace=True,axis=0)
        
        return anova
        
    def print_anova_report(self):
        anova = self.anova_report()
        
        for i in range(len(anova)):
            if (type(anova['F-Test'][i])=='object'):
                anova['F-Test'][i] = round(anova['F-Test'][i],4)
            anova['P-Value'][i] = "%.4e" % anova['P-Value'][i]
            
        print '----------------------------------------------------------------'
        print 'One Way ANOVA Report for Analysis: '+str(self.name)
        print
        print anova
        print '----------------------------------------------------------------'
        
        return None
        
    def create_plots(self,max_vars=-1):
        a = self.anova_report()        
        
        count = 0
        for v in a.index:
            if a['Var Type'].ix[v]=='numeric':
                self.num_vars[v].plot(v)
                self.num_vars[v].plot_bubble(str(v+' - Bubble'))            
            else:
                self.char_vars[v].plot(v)
            
            count += 1
            if (max_vars!=-1 and count>=max_vars):
                return





# Returns variable type for a series, even if it has numbers stored as strings
def parse_var(series):
    if (len(series.unique())<=1):
        return 'unary'
    elif ((series.dtype == 'int64') or (series.dtype == 'float64')):
        return 'numeric'
    else:
        i=0
        while i< len(series):
            try:
                float(series[i])
                i+=1
            except:
                return 'character'
                exit
        return 'obj - all numbers'
            
'''
n1 = pd.Series(['5','4','3','334','4.0003','.0006',np.nan,np.nan])
n2 = pd.Series(['5','4','3',.0006,'cats','dogs','hamburglers',None,None])
n3 = pd.Series(['5','4','3','334','4.0003','.0006',9,30,.44308,None,None])
n4 = pd.Series([5,4,3,334,4.0003,.0006,None,None])
        
print parse_var(n1)
print parse_var(n2)
print parse_var(n3)
print parse_var(n4)

for i in test.columns:
    print i +'   -   '+ parse_var(test[i])
'''  

# Uses the parse_var function to return a tuple of lists with a df columns seperated by type    
def parse_dataframe(df,cols=None):
    num=[]
    char=[]
    unary=[]
    if cols==None:
        cols=df.columns
    for i in cols:
        t = parse_var(df[i])
        if t in ('numeric','obj - all numbers'):
            num.append(i)
        if t=='character':
            char.append(i)
        if t=='unary':
            unary.append(i)
    return (num,char,unary)
    
    
    
'''
This function buckets a numeric var into evenly sized groups, as closely as possible
keeping each unique value in a single group.  It returns a data frame with min 
and max value for each bucket.  Note there may be less than the desired number 
of bins if there are many repeated values in the series, and there will be gaps 
between bucket ranges (because it's based on observed values).
'''
def num_bucket_assign(data,var,bins):
                
    #find all unique values (Non-Missing) and sort in ascending sequence
    temp = data[[var]].loc[data[var].notnull()]
    
    
    try:
        temp['qcut_bins'] = pd.qcut(temp[var],10)
        
        v = temp['qcut_bins'].unique()
        
        #create data frame to include bin thresholds
        b = pd.DataFrame(index=range(0,len(v)),columns=['bin'])
        
        for i in range(0,len(v)):
            b['bin'].iloc[i]=(min(temp[var].loc[temp['qcut_bins']==v[i]]),
                              max(temp[var].loc[temp['qcut_bins']==v[i]])) 
        
        bin_cuts=b.sort(columns='bin').reset_index(drop=True)        
        #print "           - qucut worked"
        
    except:    
        #print "           - qucut failed"
        u = temp[var].unique()

        u.sort()
        
        #create data frame to include unique values, counts, and assigned buckets
        uni = pd.DataFrame(index=range(0,len(u)),columns=['value','count','bucket'])
        
        #determine witdh of bucket based on number of obs
        width = len(temp)/bins
        
        #initialize current bucket and bucket size count for loop
        csize=0
        bucket=0
        
        for i in range(0,len(u)):
            #get unique value and count
            uni['value'].iloc[i]=u[i]
            uni['count'].iloc[i]=len(temp.loc[temp[var] == u[i]])
    
            #update size for current bucket and assign bucket value
            csize=csize+uni['count'].iloc[i]
            uni['bucket'].iloc[i]=bucket
    
            #if current bucket size has exceeded the desired width, then start next bucket
            if csize>=width:
                csize=0
                bucket=bucket+1
        
        #merge small last bins into previous bin
        if (np.sum(uni['count'][uni.bucket==max(uni['bucket'])])<(width/3)):
            uni['bucket'].loc[uni['bucket']==max(uni['bucket'])]=max(uni['bucket'])-1
        
        #create data frame to include bin thresholds
        bin_cuts = pd.DataFrame(index=range(0,max(uni['bucket'])+1),columns=['bin'])
        
        #assign min an max bucket values
        for i in range(0,max(uni['bucket'])+1):
            bin_cuts['bin'].iloc[i]=(min(uni.loc[uni['bucket']==i,'value']),max(uni.loc[uni['bucket']==i,'value'])) 
        
    #add bucket if missing values exist
    if sum(pd.isnull(data[var]))>0:
        bin_cuts.loc[len(bin_cuts)] = 'MISS'
        
    #add bucket if more than 1% of non missing values are =0
    if (len(temp[var].loc[temp[var]==0])/float(len(temp)))>.01:
        bin_cuts.loc[len(bin_cuts)] = 'ZERO'
       
    return bin_cuts


'''
This function calculates some stats for a series of numeric values, using 
the above function to groups the series into approx equal sized groups
'''
def num_bin_stats(var,target,buckets=10,target_type='binary'):
    
    #create a temporary series to work with
    temp = pd.DataFrame()
    temp['var']=var
    temp['target']=target
    
    #create summary dataframe for output
    if target_type == 'binary':
        cutpoints = pd.DataFrame(columns=('bin','Mean','Count','N_Target','Miss_Target','Target_Rate'))
    elif target_type == 'continuous':
        cutpoints = pd.DataFrame(columns=('bin','Mean','Count','Miss_Target','Mean_Target'))
    else:
        raise TypeError("Not a Valid Target Type (needs to be 'binary' or 'continuous')")
     
   
    #test to see if fewer levels than buckets
    if len(temp['var'].unique())>buckets:        
        granular=1
        
        #create group buckets    
        cutpoints['bin'] = num_bucket_assign(temp,'var',buckets)
        
    else:        
        granular=0
        
        #create group buckets
        cutpoints['bin'] = list(temp['var'].unique())
        cutpoints['bin'].loc[cutpoints['bin'].isnull()]='MISS'
        cutpoints.sort(columns='bin',inplace=True)
        cutpoints.reset_index(drop=True,inplace=True)

    for i in range(0,len(cutpoints)):
        
        #create series of values within bucket of interest
        if cutpoints['bin'][i]=='MISS':
            temp2 = temp.loc[temp['var'].isnull()]
        elif cutpoints['bin'][i]=='ZERO':
            temp2 = temp.loc[temp['var']==0]
        else:
            if granular==1:
                temp2 = temp.loc[(temp['var']>=cutpoints['bin'].ix[i][0]) & (temp['var']<=cutpoints['bin'].ix[i][1])]
            else:
                temp2 = temp.loc[temp['var']==cutpoints['bin'][i]]
        
        #calculate key stats
        cutpoints['Count'].iloc[i] = len(temp2)
        cutpoints['Mean'].iloc[i] = np.mean(temp2['var'])
        cutpoints['Miss_Target'].iloc[i] = sum(pd.isnull(temp2['target']))
        
        if target_type == 'binary':
            cutpoints['N_Target'].iloc[i] = sum(temp2['target'])
            cutpoints['Target_Rate'].iloc[i] = cutpoints['N_Target'].iloc[i]/float(cutpoints['Count'].iloc[i])
        
        if target_type == 'continuous':
            cutpoints['Mean_Target'].iloc[i] = np.mean(temp2['target'])
    
    return cutpoints
    


'''
This function plots a univariate view of a categorical variable, including
distribution counts and target rate within value.
'''
def stats_char(var,target,target_type='binary'):
    
    #create a temporary series to work with
    temp = pd.DataFrame()
    temp['var']=var
    temp['target']=target
    vals=temp['var'].unique()
    
    #create summary dataframe for output
    if target_type == 'binary':
        ds=pd.DataFrame(index=range(0,len(vals)),columns=['Value','Count','Missing Target','N_Target','Target Rate'])
    elif target_type == 'continuous':
        ds=pd.DataFrame(index=range(0,len(vals)),columns=['Value','Count','Missing Target','Mean_Target'])
    else:
        raise TypeError("Not a Valid Target Type (needs to be 'binary' or 'continuous')")
   
    ds['Value']=vals
    
    #fill in counts and target rate values
    for i in range(0,len(ds)):
        if (str(ds['Value'][i])=='nan'):
            temp2=temp.loc[temp['var'].isnull()]
        else:
            temp2=temp.loc[temp['var']==ds['Value'][i]]
            
        ds['Count'][i]=len(temp2)
        ds['Missing Target'][i]=sum(temp2['target'].isnull())
        if target_type == 'binary':
            ds['N_Target'][i]=sum(temp2['target'])
            ds['Target Rate'][i]=np.mean(temp2['target'])
        if target_type == 'continuous':
            ds['Mean_Target'][i]=np.mean(temp2['target'])

    ds.sort(columns='Value',inplace=True)
    ds.reset_index(drop=True,inplace=True)
    
    return ds
    
    
'''
This function takes binned #'s of Goods and Bads and returns Information Value (IV).
Input should be in the form of a pandas data frame with total and target #'s.
'''
def info_value(df,total,bads):
    
    df2=df[[total,bads]]
    
    #dont calculate IV if there are more than 20 levels ... not accurate
    if len(df2)>20:
        return '0-888 Too Many Bins (>20)'
    
    else:
        df2['goods']=df2[total]-df2[bads]
        
        #dont calculate IV if any buckets are empty ... not accurate
        for i in range(0,len(df2)):
            if (df2['goods'][i]==0 or df2[bads][i]==0):
                return '0-999 Bins With No Events'
                exit
                
        df2['good_pct']=df2['goods']/sum(df2['goods'])
        df2['bad_pct']=df2[bads]/sum(df2[bads])
        df2['a']=(df2['good_pct']-df2['bad_pct'])
        df2['b']=(df2['good_pct']/df2['bad_pct'])
        df2['c']=np.log(pd.Series(df2['b'],dtype=float))
        df2['iv_in']=df2['a']*df2['c']
        return str(sum(df2['iv_in']))

'''   
st=num_bin_stats(snap['Years In Business'],snap['target'],10)
info_value(st,'Count','N_Target')
'''

'''
These functions takes a predictor variable, a target variable, and for numberic vars
it expects predictor variable bins, and returns an F-test based on using the target 
avaerage in each bin.  The idea is to provide a variable screening vehicle for 
continuous targets.
'''

def F_test_num(var,target,bins):
    bins = bins[bins.bin != 'ZERO']
    bins = bins[bins.bin != 'MISS']
    temp = pd.concat([var,target],axis=1)
    if (type(bins['bin'][0])==tuple):
        cuts = []
        for i in range(0,len(bins)):
            cuts.append(bins['bin'].iloc[i][0])
        cuts.append(bins['bin'].iloc[len(bins)-1][1]+.01)
        temp['cat'] = list(pd.cut(temp[temp.columns[0]],bins=cuts,labels=bins['bin']))   
        ins = []
        for i in range(0,len(bins)):
            g = temp[temp.columns[1]].loc[temp['cat']==bins['bin'][i]].dropna()
            if (len(g>0)):
                ins.append(g)
    else:
        ins = []
        for i in range(0,len(bins)):
            g = temp[temp.columns[1]].loc[temp[temp.columns[0]]==bins['bin'][i]].dropna()
            if (len(g>0)):
                ins.append(g)        
    results = st.f_oneway(*ins)
    return results

def F_test_char(var,target):
    temp = pd.concat([var,target],axis=1)
    uni = temp[temp.columns[0]].unique()
    if (len(uni)>20):
        return ('NA - Too Many Values (>20)',1)
    else:
        ins = []
        for i in range(0,len(uni)):
            g = temp[temp.columns[1]].loc[temp[temp.columns[0]]==uni[i]].dropna()
            if (len(g)>0):
                ins.append(g)
        results = st.f_oneway(*ins)
        return results
    