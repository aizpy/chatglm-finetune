from peft import LoraConfig, get_peft_model, TaskType

def load_lora_config(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"])
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
