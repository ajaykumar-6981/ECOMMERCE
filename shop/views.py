from django.shortcuts import render
from .models import Product,Contact

from math import ceil

# Create your views here.
def home(request):
     allProds=[]

     catprods= Product.objects.values('category','id')
     cats={item['category'] for item in catprods}
     for cat in cats:
          prod=Product.objects.filter(category=cat)
          n= len(prod)
          nSlides = n // 4 + ceil((n / 4) - (n // 4))
          print(nSlides)
          # params = {'no_of_slides': nSlides, 'range': range(nSlides), 'product': products}
          # allProds = [[products, range(1, nSlides), nSlides],params={'allProds':allProds}
          allProds.append(([prod,range(1,nSlides),nSlides]))
     params={'allProds':allProds}
     return render(request,'home.html',params)

def about(request):
     return render(request,'about.html')

def contact(request):
     thank = False
     if request.method == "POST":
          name = request.POST.get('name', '')
          email = request.POST.get('email', '')
          phone = request.POST.get('phone', '')
          desc = request.POST.get('desc', '')
          contact = Contact(name=name, email=email, phone=phone, desc=desc)
          contact.save()
          thank = True
     return render(request,'contact.html',{'thank':thank})

def tracker(request):
     return render(request,'tracker.html')

def productView(request, myid):

    # Fetch the product using the id
    product = Product.objects.filter(id=myid)
    print(product)
    return render(request, 'shop/productView.html',{'product':product})


def checkout(request):
     return render(request,'checkout.html')


def Linear(request):
     return render(request,'linear.html')