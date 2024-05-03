from django.shortcuts import render
from .models import Blogpost

# Create your views here.
def blog(request):
     return render(request,'index.html')

def blogpost(request):
     post=Blogpost.objects.all()
     print(post)
     return render(request,'blogpost.html',{'post':post})

