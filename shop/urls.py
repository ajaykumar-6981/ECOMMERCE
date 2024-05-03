
from django.contrib import admin
from django.urls import path,include
from shop import views

urlpatterns = [
    path('',views.home,name='home'),
    path('about/',views.about,name='about'),
    path('contact/',views.contact,name='contact'),
    path('tracker/',views.tracker,name='tracker'),
    path('shop/products/<int:myid>',views.productView,name="ProductView"),
    path("checkout/",views.checkout, name="product_view"),
    path("linear/",views.Linear,name="linear")
]
