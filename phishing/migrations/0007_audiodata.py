# Generated by Django 5.0 on 2023-12-12 06:14

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("phishing", "0006_alter_mfcc_mfcc"),
    ]

    operations = [
        migrations.CreateModel(
            name="AudioData",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                (
                    "mel_np",
                    django.contrib.postgres.fields.ArrayField(
                        base_field=models.FloatField(), size=None
                    ),
                ),
                (
                    "mfcc",
                    django.contrib.postgres.fields.ArrayField(
                        base_field=models.FloatField(), size=None
                    ),
                ),
            ],
            options={
                "db_table": "audiodata",
            },
        ),
    ]