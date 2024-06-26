# Generated by Django 5.0 on 2023-12-12 01:11

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("phishing", "0003_callhistory_audio_file_alter_callhistory_contents"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="mel_spectrograms",
            name="mel_spctrograms",
        ),
        migrations.AddField(
            model_name="mel_spectrograms",
            name="mel_img",
            field=models.ImageField(null=True, upload_to="mel_spectrograms_img/"),
        ),
        migrations.AddField(
            model_name="mel_spectrograms",
            name="mel_np",
            field=models.FileField(null=True, upload_to="mel_spectrogram_np/"),
        ),
    ]
