# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:00:30 2015

@author: Jeremy
"""



import pandas as pd
import univariate_classes as UV
import time
import numpy as np



tester=UV.Univariate_Continuous(name='Univariate Test Continuous',
                  df=.....,
                  target='TotalSuppAmt',
                  exclude=['Unnamed: 0', 'WORK_ITEM_PK', 'SUPPLEMENT_FLAG'],
                  numeric_bins=10)
                  

tester=UV.Univariate_Binary(name='Univariate Test - Binary',
                  df=.....,
                  target='target',
                  exclude=['DBA','ISO','City','Payback Amount'],
                  numeric_bins=10)
                 
d =tester.bivar('Industry','Classification')
d2=tester.bivar('Industry','State')
d3=tester.bivar('Years In Business','Claification')
d4=tester.bivar('Years In Business','Total Commission')
d4=tester.bivar('Industry','Years In Business')
  
tester.print_IV_report()
tester.char_vars['Industry'].plot()
          
