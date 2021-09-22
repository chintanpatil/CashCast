from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

# Create your views here.
def home(request):
	return(render(request, 'home.html'))

def SMA(request):
	import numpy as np
	numListVal = request.POST['numList']
	numListVal = numListVal.split(',')
	numListVal = [int(i) for i in numListVal]
	windowVal = int(request.POST['window'])
	res = np.round(np.convolve(numListVal, np.ones(windowVal), 'valid')/windowVal,2)
	return render(request, "result.html", {'result':res})

def EMA(request):
	import numpy as np
	import pandas as pd
	numListVal = request.POST['numList']
	numListVal = numListVal.split(',')
	numListVal = [int(i) for i in numListVal]
	smoothVal = float(request.POST['smooth'])
	spanVal = (2-smoothVal)/smoothVal 
	numListVal = pd.Series(numListVal)
	res = list(np.round((numListVal).ewm(span = spanVal, adjust=False).mean(),2))
	return render(request, "result.html", {'result':res})


def SFA(request):
	if request.method == 'POST':
		
		import pandas as pd
		import numpy as np
		import datetime
		from datetime import timedelta as td
		import calendar
		from dateutil.relativedelta import relativedelta
		
		uploaded_file = request.FILES['csv_file']
		df = pd.read_csv(uploaded_file)
		df[['Invoice Date','Paid Date']] = df[['Invoice Date','Paid Date']].apply(pd.to_datetime, format='%m/%d/%Y')
		df['Due Date'] = df['Invoice Date'] + td(days=30)


		print(df)

		def ARaging(ARlog, calcDate):
		    ################################################
		    '''Function takes the ARlog of a customer and returns AR aging report as it would have looked on calcDate'''
		    
		    ARaged = ARlog.loc[(ARlog['Invoice Date']<calcDate) & (ARlog['Paid Date']>calcDate)].copy() # all invoices issued before date "calcDate" and not paid until date "calcDate". 
		    ARaged.loc[:,'Days Remaining']=(ARaged['Due Date'] - calcDate).dt.days

		    # create a list of our conditions
		    conditions = [
		        (ARaged.loc[:,'Days Remaining'] <= 0) & (ARaged.loc[:,'Days Remaining'] >= -30),
		        (ARaged.loc[:,'Days Remaining'] < -30) & (ARaged.loc[:,'Days Remaining'] >= -60),
		        (ARaged.loc[:,'Days Remaining'] < -60) & (ARaged.loc[:,'Days Remaining'] >= -90),
		        (ARaged.loc[:,'Days Remaining'] < -90) & (ARaged.loc[:,'Days Remaining'] >= -120),
		        (ARaged.loc[:,'Days Remaining'] < -120), # if payment is overdue 120 days then it is put under Bad Debt
		        (ARaged.loc[:,'Days Remaining'] >= 0), # represents current invoices. Invoices that are not due yet
		        ]

		    # list of the values we want to assign for each condition
		    values = ['1','2','3','4','5','0']

		    # create a new column and assign values to it using the lists "conditions" and "values" as arguments
		    ARaged.loc[:,'status'] = np.select(conditions, values)
		    ARaged = ARaged.groupby(by=['status']).sum().reset_index()
		    del(ARaged['Days Remaining'])

		    missing = list(set(['0','1','2','3','4','5']) - set(ARaged.status.drop_duplicates()))
		    missing = [int(ele) for ele in missing]; missing.sort()
		    missing = [list(ele) for ele in zip(missing,[0]*len(missing))] 
		    missing = pd.DataFrame(missing,columns=['status','Amount'])

		    ARaged = ARaged.append(missing)
		    ARaged.loc[:,'status'] = ARaged['status'].astype(int) 
		    ARaged = ARaged.sort_values(by=['status']).reset_index(); del(ARaged['index']); del(ARaged['status'])
		    ARaged = ARaged.T
		    ARaged['calcDate'] = calcDate
		    return(ARaged)


		def reporting(df_uploaded):
		    GrandARaging = pd.DataFrame([])
		    for cust in df_uploaded['Customer'].drop_duplicates():
		        df_uploaded_cust = df_uploaded[df_uploaded['Customer']==cust].copy()

		        i = min(df_uploaded_cust['Invoice Date'])
		        i = i + relativedelta(day=31)

		        #counter = 0
		        while i < max(df_uploaded_cust['Invoice Date']):
		            #counter = counter  + 1 
		            #print(i)
		            i = i + relativedelta(months=+1)
		            i = i + relativedelta(day=31)
		            temp1 = ARaging(df_uploaded_cust,i); temp1['customer'] = cust
		            
		            ### Calculating Payments Recieved in the month at the end of which AR aging calculations are made.### 
		            j = i - relativedelta(months=+1)
		            j = j - relativedelta(day=31)
		            temp2 = df_uploaded_cust.loc[(df_uploaded_cust['Paid Date']<i) & (df_uploaded_cust['Paid Date']>j) ].copy() ## dataframe of all payments made in month of calcDate
		            temp1['ActualCashFlowIn'] = sum(temp2['Amount'])
		            #####################################################################################################

		            GrandARaging = GrandARaging.append(temp1)
		            #GrandARaging = GrandARaging.reset_index()
		            del(temp1); del(temp2)

		    GrandARaging = GrandARaging.reset_index()
		    del(GrandARaging['index'])
		    return(GrandARaging)


		def calcProbs(ARagingReport):
			s1ToP = []; s2ToP = [];s3ToP = []; s4ToP = []
		    	
			s1ToP = (ARagingReport[1] - ARagingReport[2].shift(-1))/ARagingReport[1]
			s2ToP = (ARagingReport[2] - ARagingReport[3].shift(-1))/ARagingReport[2]
			s3ToP = (ARagingReport[3] - ARagingReport[4].shift(-1))/ARagingReport[3]
			s4ToP = (ARagingReport[4] - ARagingReport[5].shift(-1))/ARagingReport[4]
		    
			return(s1ToP,s2ToP,s3ToP,s4ToP)



		def forecast(df_uploaded):
		    GrandARaging = reporting(df)
		    aggGrandARaging = GrandARaging.groupby('calcDate').sum()

		    # s1ToP, s2ToP,..,s4ToP are equaivalent to T1, T2,...,T4 in Corcoran
		    s1ToP, s2ToP, s3ToP, s4ToP = calcProbs(aggGrandARaging)
		    aggGrandARaging['s1ToP'] = s1ToP.shift(1)
		    aggGrandARaging['s2ToP'] = s2ToP.shift(1)
		    aggGrandARaging['s3ToP'] = s3ToP.shift(1)
		    aggGrandARaging['s4ToP'] = s4ToP.shift(1)
		    aggGrandARaging = aggGrandARaging.reset_index()
		    
		    # We assume stage 5 is bad payments. I.e., once account payable goes to stage 5, no payment is coming back 
		    # Here, we are calculating exponential forecasts of transition probabilities
		    aggGrandARaging['expProb_s1ToP'] = aggGrandARaging['s1ToP'].ewm(alpha=0.9, adjust=False).mean()
		    aggGrandARaging['expProb_s2ToP'] = aggGrandARaging['s2ToP'].ewm(alpha=0.9, adjust=False).mean()
		    aggGrandARaging['expProb_s3ToP'] = aggGrandARaging['s3ToP'].ewm(alpha=0.9, adjust=False).mean()
		    aggGrandARaging['expProb_s4ToP'] = aggGrandARaging['s4ToP'].ewm(alpha=0.9, adjust=False).mean()
		    aggGrandARaging['expProb_s5ToP'] = 0*len(aggGrandARaging)
		    
		    # Here we are using exponential smoothing to forecast the amount of payments that would go from stage i to P in time period t
		    aggGrandARaging['fore1'] = aggGrandARaging[1]*aggGrandARaging['expProb_s1ToP']
		    aggGrandARaging['fore2'] = aggGrandARaging[1]*aggGrandARaging['expProb_s2ToP']
		    aggGrandARaging['fore3'] = aggGrandARaging[1]*aggGrandARaging['expProb_s3ToP']
		    aggGrandARaging['fore4'] = aggGrandARaging[1]*aggGrandARaging['expProb_s4ToP']
		    aggGrandARaging['fore5'] = aggGrandARaging[1]*aggGrandARaging['expProb_s5ToP']
		    aggGrandARaging['stochasticForecast'] = aggGrandARaging['fore1']+aggGrandARaging['fore2']+aggGrandARaging['fore3']+aggGrandARaging['fore4']+aggGrandARaging['fore5']
		    return(aggGrandARaging)


		def shapeOfYou(df_on_calcDate):
		    PLT = df_on_calcDate['Paid Date'] - df_on_calcDate['Invoice Date']
		    xbar = PLT.mean().days
		    gamma = PLT.min().days
		    shape = np.round(2*(xbar-gamma)/(np.pi)**0.5,4)
		    return(shape,gamma)


		 ## Calculate payment probability
		def calcPayProb(df_monthly,shapeVal,gammaVal):
		    probPaid = []
		    for j in range(0,len(df_monthly)):
		        if df_monthly[j:j+1]['Invoice Sent days ago'].dt.days.item()>= gammaVal:
		            pp = np.round(1 - np.exp(-(2*(df_monthly['Invoice Sent days ago'][j:j+1].dt.days.item()-gammaVal)*df_monthly['delta t'][j:j+1].item()+df_monthly['delta t'][j:j+1].item()**2)/shapeVal**2),4)
		        elif (df_monthly[j:j+1]['Invoice Sent days ago'].dt.days.item()< gammaVal) and (df_monthly[j:j+1]['Invoice Sent days ago'].dt.days.item() + df_monthly[j:j+1]['delta t'].item()>= gammaVal):
		            pp = np.round(1 - np.exp(-((df_monthly['Invoice Sent days ago'][j:j+1].dt.days.item() + df_monthly['delta t'][j:j+1].item()-gammaVal)**2)/shapeVal**2),4)
		        elif df_monthly[j:j+1]['Invoice Sent days ago'].dt.days.item() + df_monthly[j:j+1]['delta t'].item()<= gammaVal:
		            pp = 0 
		        probPaid.append(pp)
		    return(probPaid)

		def bayForecast(df_uploaded):
		    custList = []
		    calcDateList = []
		    forecastList = []
		    bayesianForecast = pd.DataFrame([])
		    for cust in df_uploaded['Customer'].drop_duplicates():
		        df_uploaded_cust = df_uploaded[df_uploaded['Customer']==cust].copy()
		        
		        i = min(df_uploaded_cust['Invoice Date'])
		        i = i + relativedelta(day=31)
		        
		        #counter = 0
		        while i < max(df_uploaded_cust['Invoice Date']):
		            #counter = counter  + 1 
		            #print(i)
		            i = i + relativedelta(months=+1)
		            i = i + relativedelta(day=31)
		            temp = df_uploaded_cust.loc[(df_uploaded_cust['Invoice Date']<i)].copy()
		            shape,gamma = shapeOfYou(temp)

		            temp1 = temp[temp['Paid Date']>i].copy(); temp1.reset_index(drop=True) #Keep only the invoices not cleared till calcDate, i. Forecast will be made only for these invoices
		            temp1['Invoice Sent days ago'] = i-temp1['Invoice Date']
		            temp1['delta t'] = [30]*len(temp1)

		            custList.append(cust)
		            calcDateList.append(i)
		            forecastList.append(sum(temp1['Amount']*calcPayProb(temp1,shape,gamma))) #Dollars coming in
		    
		    bayesianForecast['Customer'] = custList
		    bayesianForecast['calcDate'] = calcDateList
		    bayesianForecast['bayForecast'] = forecastList
		    bayesianForecast = bayesianForecast.groupby('calcDate').sum().reset_index()
		    return(bayesianForecast)



	result_corcoran = forecast(df)
	result_pate = bayForecast(df)
	result = result_corcoran.merge(result_pate,on='calcDate')
	result = result[['calcDate','ActualCashFlowIn','stochasticForecast','bayForecast']]

	beta = 0.3
	result['Forecast'] = beta*result['stochasticForecast'] + (1-beta)*result['bayForecast']
	result
	#result = result.to_html()

	#result_corcoran = result_corcoran[['ActualCashFlowIn','fore_inflow','calcDate']]
	result = result.dropna()

	#months = ['1','2','3']
	#forecast_cash = [nan,23,21]
	months = result['calcDate'].astype(str).tolist()
	stochasticForecast = np.round(result['stochasticForecast'].astype(float),2).tolist()
	bayForecast = np.round(result['bayForecast'].astype(float),2).tolist()
	forecast = np.round(result['Forecast'].astype(float),2).tolist()
	actual_cash = np.round(result['ActualCashFlowIn'].astype(float),2).tolist()
	context = {'months':months, 'forecast':forecast, 'actual_cash':actual_cash, 'stochasticForecast':stochasticForecast, 'bayForecast':bayForecast}

	#return HttpResponse(result)
	#return render(request, 'result.html', {'result':GrandARaging})
	return render(request, 'viz.html', context) 


@login_required
def dashboard(request):
	return(render(request, 'dashboard.html'))