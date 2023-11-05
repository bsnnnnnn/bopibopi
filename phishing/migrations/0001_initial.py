# Generated by Django 4.2.7 on 2023-11-05 11:41

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Organizations",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=20)),
                ("image", models.CharField(max_length=255)),
                ("number", models.CharField(max_length=50)),
                ("work", models.CharField(max_length=100)),
                ("url", models.CharField(max_length=255)),
                ("cat", models.CharField(max_length=20)),
            ],
        ),
    ]