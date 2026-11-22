"""API Gateway -> Lambda -> SageMaker endpoint -> DynamoDB (+ SNS reminder).

Environment variables to set on the function:
  ENDPOINT_NAME   wellbeing-endpoint
  TABLE_NAME      WellbeingLogs   (partition key student_id: String, sort key date: String)
  TOPIC_ARN       arn of the SNS topic (optional)
  LOW_SCORE       50
Role needs: sagemaker:InvokeEndpoint, dynamodb:PutItem/Query on the table, sns:Publish on the topic.
Enable CORS on the API Gateway method so the page can call it.
"""
import json, os
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

runtime = boto3.client("sagemaker-runtime")
table = boto3.resource("dynamodb").Table(os.environ.get("TABLE_NAME", "WellbeingLogs"))
sns = boto3.client("sns")
FIELDS = {"sleep": (0, 14), "study": (0, 16), "screen": (0, 18), "meals": (0, 6), "activity": (0, 300), "mood": (1, 5)}
HEADERS = {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*",
           "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "POST,OPTIONS"}


def reply(code, body):
    return {"statusCode": code, "headers": HEADERS, "body": json.dumps(body)}


def lambda_handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return reply(200, {})
    try:
        body = json.loads(event.get("body") or "{}")
        student = str(body["student_id"])[:40]
        date = str(body["date"])[:10]
        vals = {}
        for k, (lo, hi) in FIELDS.items():
            v = float(body[k])
            if not lo <= v <= hi:
                return reply(400, {"error": f"{k} must be between {lo} and {hi}"})
            vals[k] = v
    except (KeyError, ValueError, TypeError):
        return reply(400, {"error": "Send student_id, date, sleep, study, screen, meals, activity, mood"})

    resp = runtime.invoke_endpoint(EndpointName=os.environ["ENDPOINT_NAME"],
                                   ContentType="application/json", Body=json.dumps(vals))
    result = json.loads(resp["Body"].read())

    table.put_item(Item={"student_id": student, "date": date, "score": result["score"],
                         **{k: Decimal(str(v)) for k, v in vals.items()}})

    recent = table.query(KeyConditionExpression=Key("student_id").eq(student),
                         ScanIndexForward=False, Limit=7)["Items"]
    history = [{"date": i["date"], "score": int(i["score"])} for i in reversed(recent)]
    low = int(os.environ.get("LOW_SCORE", 50))
    last3 = history[-3:]
    if os.environ.get("TOPIC_ARN") and len(last3) == 3 and all(h["score"] < low for h in last3):
        sns.publish(TopicArn=os.environ["TOPIC_ARN"], Subject="A gentle check-in",
                    Message="Your wellbeing score has been low for three days. A short walk or an early night "
                            "could help. If it feels heavy, please talk to someone you trust.")
    return reply(200, {"score": result["score"], "factors": result["factors"], "history": history})
