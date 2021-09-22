from django.shortcuts import render
from django.http import HttpResponse

posts = [
	{
		'author':'Chintan',
		'title':'Blog Post 1 ',
		'content': '@Rakshith, What is up?',
		'date_posted':'December 17, 2020'

		
	},
	{
		'author':'Rakshith',
		'title':'Blog Post 2 ',
		'content': 'Nothing much.',
		'date_posted':'December 17, 2020'	
	}
]

def home(request):
	context = {
		'posts':posts
	}
	return render(request,'blog/home.html', context)

def about(request):
	return render(request,'blog/about.html',{'title':'About'})