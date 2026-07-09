from django.contrib import admin
from .models import Student, AttendanceLog


class StudentAdmin(admin.ModelAdmin):
    list_display = ("student_id", "name", "major", "total_attendance")


admin.site.register(Student, StudentAdmin)
admin.site.register(AttendanceLog)