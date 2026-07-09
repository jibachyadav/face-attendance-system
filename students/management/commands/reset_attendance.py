"""
Custom Django management command to reset attendance logs for testing/demo.

Usage:
    python manage.py reset_attendance                  # deletes ALL attendance logs
    python manage.py reset_attendance --today          # deletes only today's logs
    python manage.py reset_attendance --student S001   # deletes logs for one student
    python manage.py reset_attendance --full           # also resets total_attendance to 0 for all students
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from students.models import AttendanceLog, Student


class Command(BaseCommand):
    help = "Reset attendance logs for testing/demo purposes"

    def add_arguments(self, parser):
        parser.add_argument(
            "--today", action="store_true",
            help="Only delete logs from today (instead of all logs)",
        )
        parser.add_argument(
            "--student", type=str, default=None,
            help="Only delete logs for a specific student_id",
        )
        parser.add_argument(
            "--full", action="store_true",
            help="Also reset total_attendance counter to 0 for all students",
        )

    def handle(self, *args, **options):
        qs = AttendanceLog.objects.all()

        if options["today"]:
            today = timezone.now().date()
            qs = qs.filter(timestamp__date=today)

        if options["student"]:
            try:
                student = Student.objects.get(student_id=options["student"])
            except Student.DoesNotExist:
                self.stderr.write(self.style.ERROR(f"Student {options['student']} not found."))
                return
            qs = qs.filter(student=student)

        count = qs.count()
        qs.delete()

        self.stdout.write(self.style.SUCCESS(f"Deleted {count} attendance log(s)."))

        if options["full"]:
            updated = Student.objects.update(total_attendance=0)
            self.stdout.write(self.style.SUCCESS(f"Reset total_attendance to 0 for {updated} student(s)."))
