# Generated manually for PredictionRequest model.

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="PredictionRequest",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("latitude", models.FloatField()),
                ("longitude", models.FloatField()),
                ("request_date", models.DateField()),
                ("grid_id", models.CharField(blank=True, max_length=32)),
                ("district", models.CharField(blank=True, max_length=128)),
                ("province", models.CharField(blank=True, max_length=128)),
                ("selected_crop", models.CharField(blank=True, max_length=64)),
                ("recommended_crop", models.CharField(blank=True, max_length=64)),
                ("predicted_yield_kg_ha", models.FloatField(blank=True, null=True)),
                ("confidence", models.FloatField(blank=True, null=True)),
                ("health_label", models.CharField(blank=True, max_length=32)),
                ("model_name", models.CharField(blank=True, max_length=64)),
                ("status", models.CharField(choices=[("success", "Success"), ("error", "Error")], default="success", max_length=16)),
                ("error_message", models.TextField(blank=True)),
                ("request_payload", models.JSONField(blank=True, default=dict)),
                ("response_payload", models.JSONField(blank=True, default=dict)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
