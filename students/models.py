from django.db import models


class Student(models.Model):
    student_id = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=100)
    major = models.CharField(max_length=100, blank=True, null=True)
    starting_year = models.IntegerField(blank=True, null=True)
    total_attendance = models.IntegerField(default=0)
    photo = models.ImageField(upload_to="student_photos/", blank=True, null=True)

    def __str__(self):
        return f"{self.student_id} - {self.name}"


class AttendanceLog(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="logs")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.student.student_id} @ {self.timestamp}"
