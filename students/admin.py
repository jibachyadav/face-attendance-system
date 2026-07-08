from django.contrib import admin
from .models import Student, AttendanceLog

admin.site.register(Student)
admin.site.register(AttendanceLog)