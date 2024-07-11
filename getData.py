from modeling.asr import SLAM_ASR
from datasets import load_dataset, Dataset, DatasetDict
import soundfile as sf

def format_dataset(dataset):
        def map_to_array(batch):
            speech, _ = sf.read(batch["file"])
            batch["speech"] = speech
            return batch

        dataset = dataset.map(map_to_array)
        print(dataset)
        # print(dataset.column_names)
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["speech", "text"]
            ]
        )
        return dataset

dataset = load_dataset("librispeech_asr", "clean", trust_remote_code=True)

dataset = DatasetDict({
        'train': dataset['train.360'],
        'validation': dataset['validation'],
}
)


print(f"dataset:{dataset}")

dataset = dataset.remove_columns(["file","speaker_id","chapter_id","id"])

print(f"dataset after remove:{dataset}")

print(f"dataset array:{dataset['train'][0]['audio']["array"]}")

def prepare_dataset(batch):
    batch["input_values"] = batch["audio"]["array"]
    
    del batch["audio"]
    
    return batch

dataset = dataset.map(prepare_dataset, num_proc=16)
print(f"dataset after process:{dataset}")
# print(f"dataset after remove columns:{dataset}")

# print(f"audio in dataset{dataset['train'][0]['audio']}")




# def prepare_dataset(batch,processor):
#             audio = batch["audio"]

#             # batched output is "un-batched" to ensure mapping is correct
#             batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

#             return batch


# dataset = format_dataset(dataset)

# print(f"dataset after format:{dataset}")