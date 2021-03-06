from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('sendImage/', views.SendImage.as_view(), name="SendImage"),
    path('getOutput/', views.show_output, name="getOutput"),
    path('get_output/', views.get_output, name="get_output"),
]
