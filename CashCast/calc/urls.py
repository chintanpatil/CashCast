from django.urls import path
from . import views

urlpatterns = [
	path('',views.home, name='home'),
	path('dashboard',views.dashboard, name='dashboard'),
	path('SMA',views.SMA,name='SMA'),
	path('EMA',views.EMA,name='EMA'),
	path('SFA',views.SFA,name='SFA')
]