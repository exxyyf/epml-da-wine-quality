from clearml import PipelineController


def run_pipeline():
    pipe = PipelineController(
        project="Wine Quality Prediction",
        name="wine-quality-training-pipeline",
        version="1.0.0",
    )

    pipe.set_default_execution_queue("services")

    # ===== Step: training =====
    pipe.add_step(
        name="train",
        base_task_project="Wine Quality Prediction",
        base_task_name="baseline-model-training",
        execution_queue="services",
        parameter_override={
            "Args/model_name": "rf",
            "Args/seed": 121212,
        },
    )

    pipe.start()


if __name__ == "__main__":
    run_pipeline()
