# src/ingestion/adf_trigger.py

def trigger_adf_pipeline(source_name):
    """
    Simulate triggering Azure Data Factory pipeline
    Args:
        source_name (str): One of ['emails', 'chat_logs', 'tickets']
    """
    print(f"[INFO] Triggering ADF pipeline for source: {source_name}")
    # This would normally trigger ADF using Azure SDK (omitted)
    print(f"[INFO] {source_name} data ingested into Azure Data Lake.")

if __name__ == "__main__":
    for source in ['emails', 'chat_logs', 'tickets']:
        trigger_adf_pipeline(source)

"""

#from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
import os

# Set your Azure info here
SUBSCRIPTION_ID = "your-subscription-id"
RESOURCE_GROUP = "your-resource-group"
FACTORY_NAME = "your-data-factory-name"

# List of your support pipelines
PIPELINES = {
    "emails": "ingest_emails",
    "chat_logs": "ingest_chat_logs",
    "tickets": "ingest_tickets"
}


def trigger_adf_pipeline(pipeline_name):
    
    # Trigger an Azure Data Factory pipeline using Azure SDK.

    print(f"[INFO] Triggering pipeline: {pipeline_name}")

    # Authenticate with Azure
    credentials = DefaultAzureCredential()
    adf_client = DataFactoryManagementClient(credentials, SUBSCRIPTION_ID)

    # Run the pipeline
    run_response = adf_client.pipelines.create_run(
        resource_group_name=RESOURCE_GROUP,
        factory_name=FACTORY_NAME,
        pipeline_name=pipeline_name
    )

    print(f"[SUCCESS] Pipeline '{pipeline_name}' triggered successfully. Run ID: {run_response.run_id}")


if __name__ == "__main__":
    for source, pipeline in PIPELINES.items():
        trigger_adf_pipeline(pipeline)
"""