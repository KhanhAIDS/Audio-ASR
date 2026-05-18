from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'asr_completed_events', # The topic we defined in main.py
    bootstrap_servers=['192.168.40.96:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for ASR Events from the AI...")
for message in consumer:
    print(f"\n--- NEW EVENT RECEIVED ---")
    print(f"Job ID: {message.value['job_id']}")
    print(f"Transcript: {message.value['full_transcript']}")