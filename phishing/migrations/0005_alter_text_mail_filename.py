# Generated by Django 4.2.6 on 2023-11-10 05:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("phishing", "0004_text_mail_filename"),
    ]

    operations = [
        migrations.AlterField(
            model_name="text_mail",
            name="filename",
            field=models.CharField(max_length=50, null=True),
        ),
    ]