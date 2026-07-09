import csv
from django.http import HttpResponse
from django.contrib import admin
from .models import Student, AttendanceLog


@admin.action(description="Export selected attendance logs as CSV")
def export_attendance_csv(modeladmin, request, queryset):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="attendance_logs.csv"'

    writer = csv.writer(response)
    writer.writerow(["Student ID", "Name", "Timestamp"])

    for log in queryset.select_related("student"):
        writer.writerow([log.student.student_id, log.student.name, log.timestamp])

    return response


class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = ("student", "timestamp")
    list_filter = ("timestamp",)
    search_fields = ("student__student_id", "student__name")
    actions = [export_attendance_csv]


class StudentAdmin(admin.ModelAdmin):
    list_display = ("student_id", "name", "major", "total_attendance")
    search_fields = ("student_id", "name")


admin.site.register(Student, StudentAdmin)
admin.site.register(AttendanceLog, AttendanceLogAdmin)
