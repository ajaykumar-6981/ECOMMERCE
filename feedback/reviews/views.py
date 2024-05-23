from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import ReviewForm
from .models import Review
from django.views import View
# Create your views here.


class ReviewView(View):
     def get(self, request):
         form=ReviewForm(request.POST)

         return render(request,"reviews/review.html",{
               "form":form
         })


     def post(self,request):
         form = ReviewForm(request.POST)

         if form.is_valid():
             form.save()
             return HttpResponseRedirect("thank_you")
         return render(request,"reviews/review.html",{
             "form":form
         })

#
# def review(request):
#        if request.method =='POST':
#               form=ReviewForm(request.POST)
#               if form.is_valid():
#                    review=Review(
#                        user_name=form.cleaned_data['user_name'],
#                        review_text=form.cleaned_data['review_text'],
#                        rating=form.cleaned_data['rating'])
#                    review.save()
#                    # print(username=form.cleaned_data['user_name'])
#                    return HttpResponseRedirect("/thank_you")
#
#        #        if entered_username=="":
       #               return render(request,'reviews/review.html',{
       #                      "has_error": True
       #               })
       #
       #        print(entered_username)
       #        # return render(request,"reviews/thank_you.html")
       #        return HttpResponseRedirect("/thank_you")
       #
       # return render(request,'reviews/review.html',{
       #        "has_error": False
       # })

       # form=ReviewForm()
       # return render(request, 'reviews/review.html', {
       #        "form":form
       # })

def thank_you(request):
       return render(request,'reviews/thank_you.html')