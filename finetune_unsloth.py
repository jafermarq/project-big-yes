
from logging import INFO

from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

from flwr.common.logger import log
from flwr.common import Context, Message, RecordDict, MetricRecord, ConfigRecord
from flwr.client import ClientApp


max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

# Get LAION dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"


def task(max_steps: int) -> float:
    dataset = load_dataset("json", data_files = {"train" : url}, split = "train")


    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length = 2048, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = False, # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        tokenizer = tokenizer,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = max_steps,
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            seed = 3407,
        ),
    )
    res = trainer.train()
    final_train_loss = res.training_loss
    print("Final training loss:", final_train_loss)

    return final_train_loss


# Flower ClientApp
app = ClientApp()

@app.train("finetune")
def finetune(msg: Message, context: Context):

    max_steps = msg.content["config"]["max-steps"]
    
    log(INFO, f"ClientApp starting finetuning for {max_steps = }")

    final_train_loss = task(max_steps)

    reply_content = RecordDict({"results": MetricRecord({"train-loss": final_train_loss})})

    return Message(content=reply_content, reply_to=msg)



if __name__ == "__main__":

    # Construct a Message
    max_steps = 128
    msg = Message(content=RecordDict({"config": ConfigRecord({'max-steps': max_steps})}), dst_node_id=123, message_type="train.finetune")

    # Process Message with ClientApp
    reply_message = app(message=msg, context=Context)
    final_loss = reply_message.content["results"]["train-loss"]
    log(INFO, f"Final train loss: {final_loss}")