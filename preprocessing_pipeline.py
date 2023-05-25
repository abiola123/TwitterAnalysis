import argparse
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
import re
from pyspark.conf import SparkConf
from transformers import pipeline


parser = argparse.ArgumentParser(
    description="Parse tweets JSONL file and create a regression DataFrame"
)

parser.add_argument(
    "--path", type=str, help="Path to the input JSONL file containing tweets"
)

parser.add_argument(
    "--output_name",
    type=str,
    default="american_politicians_regression_df",
    help="Name of the output CSV file",
)
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Number of worker processes to use for parallel processing",
)
parser.add_argument(
    "--annotation_threshold",
    type=int,
    default=14,
    help="Number of annotations to consider for the regression",
)

parser.add_argument(
    "--USA",
    type=bool,
    default=True,
    help="Whether to use american time or not",
)


# Load data from json file and return RDD
def load_data_set(spark, path):
    print("**Loading data from json file**")
    df = spark.read.json(path)
    print("Schema written to file\n")
    print(df.schema, file=open("schema", "a"))
    return df.rdd


def get_most_frequent_annotations(rdd, threshold):
    print("**Getting most frequent annotations** \n")
    annotations = (
        rdd.flatMap(lambda x: x["data"])
        .map(lambda x: x["context_annotations"])
        .filter(lambda x: x is not None)
        .flatMap(lambda x: list(set([y["entity"]["name"] for y in x])))
        .map(lambda x: (x, 1))
        .reduceByKey(lambda x, y: x + y)
        .sortBy(lambda x: x[1], ascending=False)
    )

    most_frequent_annotations = annotations.take(threshold)

    # remove the first one which is 'Politics' (present in nearly all tweets)
    # TODO: check if this is also the case for celebrities
    most_frequent_annotations = list(map(lambda x: x[0], most_frequent_annotations))

    annotation_dict = {
        annotation: index for index, annotation in enumerate(most_frequent_annotations)
    }

    return most_frequent_annotations, annotation_dict


def extract_relevant_fields(rdd):
    return (
        rdd.filter(lambda x: x["data"])
        .flatMap(lambda x: x["data"])
        .filter(lambda x: x["entities"])
        .filter(lambda x: x["context_annotations"] is not None)
        .map(
            lambda x: {
                "tweet_text": x["text"],
                "tweet_date": x["created_at"],
                "tweet_hashtags": x["entities"]["hashtags"],
                "tweet_mentions": x["entities"]["mentions"],
                "tweet_urls": x["entities"]["urls"],
                "user_id": x["author_id"],
                "tweet_id": x["id"],
                "context_annotations": x["context_annotations"],
                "impression_count": x["public_metrics"]["impression_count"],
            }
        )
    )


def apply_processing_pipeline(
    json_rdd,
    json_rdd_data_fields,
    most_frequent_annotations,
    annotation_dict,
    output_name,
    zero_shot_classification,
    path,
    USA,
):
    # region INNER FUNCTIONS
    def analyse_sentiment(x):
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(x)
        return vs["compound"]

    def add_key_value(x, key, value):
        x[key] = value
        return x

    def keep_medias_only(x):
        urls = x["tweet_urls"]
        if not urls:
            return []
        media_urls = [
            url["media_key"] for url in urls if "media_key" in url and url["media_key"]
        ]
        return media_urls

    def get_number_medias(x):
        return len(x["tweet_media_keys"])

    def get_number_external_urls(x):
        return (len(x["tweet_urls"]) if x["tweet_urls"] else 0) - x[
            "tweet_medias_count"
        ]

    def get_period_of_day(x, USA=True):
        date_time = datetime.strptime(x["tweet_date"], "%Y-%m-%dT%H:%M:%S.%fZ")

        # transform to american time from UTC
        if USA:
            date_time = date_time.replace(hour=(date_time.hour - 5) % 24)
        else:
            date_time = date_time.replace(hour=(date_time.hour + 1) % 24)

        return date_time.hour

    def one_hot_encoding(x, encoding_dict):
        encoding = [0] * len(encoding_dict)

        annotations = x["context_annotations"]

        if not annotations:
            return encoding

        for annotation in annotations:
            if isinstance(annotation, str):
                name = annotation

            elif not annotation["entity"]:
                continue

            else:
                name = annotation["entity"]["name"]

            if name in encoding_dict:
                encoding[encoding_dict[name]] = 1

        return encoding

    def add_dummy_encoding(x, column_names):
        encoding = dict(zip(column_names, x["encoded_annotations"]))

        for key, value in encoding.items():
            cleaned = re.sub("[^A-Za-z0-9_]+", "", key.lower())
            cleaned = re.sub("__", "_", cleaned)

            x[f'dummy_{"_".join(cleaned.split(" "))}'] = value

        return x

    # endregion

    # region PROCESSING PIPELINE

    # map context annotations to clusters
    def group_context_annotations(x, most_frequent_annotations, annotation_dict):
        # first we see if the annotations of our tweet and the most frequent annotations overlap
        annotation_set = set([y["entity"]["name"] for y in x["context_annotations"]])

        intersection = annotation_set.intersection(most_frequent_annotations)

        if len(intersection) > 0:
            x["context_annotations"] = list(intersection)

        else:
            # zero shot here

            prepared_text = " ".join(annotation_set)

            zero_shot_labels = zero_shot_classification(
                prepared_text, most_frequent_annotations
            )

            x["context_annotations"] = zero_shot_labels

        return x

    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: group_context_annotations(
            x, most_frequent_annotations, annotation_dict
        )
    )

    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(
            x, "tweet_sentiment", analyse_sentiment(x["tweet_text"])
        )
    )

    # adding the tweet length to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(x, "tweet_length", len(x["tweet_text"]))
    )

    # adding the number of hashtags to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(
            x, "hashtags_count", len(x["tweet_hashtags"]) if x["tweet_hashtags"] else 0
        )
    )

    # adding the number of mentions to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(
            x, "mentions_count", len(x["tweet_mentions"]) if x["tweet_mentions"] else 0
        )
    )

    # adding the media url's only to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(x, "tweet_media_keys", keep_medias_only(x))
    )

    # adding the number of medias to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(x, "tweet_medias_count", get_number_medias(x))
    )

    # adding the number of external urls to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(
            x, "tweet_external_urls_count", get_number_external_urls(x)
        )
    )

    # adding the period of the day to the data
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(x, "tweet_period", get_period_of_day(x, USA))
    )

    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: {
            k: v
            for k, v in x.items()
            if k
            not in [
                "tweet_mentions",
                "tweet_urls",
                "tweet_hashtags",
                "tweets_media_count",
            ]
        }
    )

    # getting the annotations and putting them in clusters using one hot encoding
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_key_value(
            x, "encoded_annotations", one_hot_encoding(x, annotation_dict)
        )
    )

    # GENERATE DUMMY VARIABLES FOR CATEGORICAL VARIABLES

    # add dummy variables after one hot encoding
    json_rdd_data_fields = json_rdd_data_fields.map(
        lambda x: add_dummy_encoding(x, most_frequent_annotations)
    )

    # json_rdd_data_fields = json_rdd_data_fields.map(lambda x: add_dummy_tweet_period(x))

    # 2. Create a dataframe from the rdd

    # endregion

    print("** Applying processing pipeline (this may take some time...) **")
    regression_df = (
        json_rdd_data_fields.toDF()
        .drop(
            "context_annotations",
            "encoded_annotations",
            "tweet_date",
            "tweet_media_keys",
            # "tweet_period",
        )
        .persist()
    )
    # getting the followers count data
    json_rdd_followers = json_rdd.filter(
        lambda x: x["includes"] and x["data"] and x["includes"]["users"]
    ).map(
        lambda x: {
            "followers_count": x["includes"]["users"][0]["public_metrics"][
                "followers_count"
            ],
            "user_id": x["includes"]["users"][0]["id"],
        }
    )

    # converting the rdd to a dataframe
    json_followers_df = json_rdd_followers.toDF()
    json_followers_df = json_followers_df.dropDuplicates(["user_id"])

    # 4. Join the two dataframes in order to get the media type for each tweet.
    # If a tweet has multiple media, we will get multiple rows for the same tweet id in the exploded dataframe
    # Then when performing the join, we will have a set of media types for each tweet id in the json_medias_per_tweet_df dataframe

    regression_df = (
        regression_df.join(json_followers_df, on="user_id", how="inner")
        .drop("user_id", "tweet_id")
        .persist()
    )

    regression_df_pd = regression_df.toPandas()
    regression_df_pd.to_csv(f"{output_name}.csv", index=False)

    print(f"Processed dataframe saved to {output_name}.csv \n")

    return regression_df_pd


def pre_processing_pipeline(path, output_name, workers, annotation_threshold, USA):
    spark = (
        SparkSession.builder.appName("tweet_loader")
        .master(f"local[{workers}]")
        .config("spark.driver.memory", "10g")
        .getOrCreate()
    )

    print("**SparkContext created**")
    print(f"GUI: {spark.sparkContext.uiWebUrl}")
    print(f"AppName: {spark.sparkContext.appName}\n")

    rdd = load_data_set(spark, path)

    most_frequent_annotations, annotation_dict = get_most_frequent_annotations(
        rdd, annotation_threshold
    )

    print(f"Most frequent annotations: {most_frequent_annotations}\n")

    print(f"Annotation dictionary: {annotation_dict}\n")

    rdd_subset = extract_relevant_fields(rdd)

    pipe = spark.sparkContext.broadcast(pipeline(model="valhalla/distilbart-mnli-12-9"))

    def zero_shot_classification(text, labels):
        resp = pipe.value(text, labels, multi_label=False)

        labels_scores = zip(resp["labels"], resp["scores"])

        labels_scores = filter(lambda x: x[1] > 0.3, labels_scores)

        predicted_labels = list(map(lambda x: x[0], labels_scores))

        if len(predicted_labels) > 0:
            return predicted_labels
        else:
            return [resp["labels"][0]]

    regression_df = apply_processing_pipeline(
        rdd,
        rdd_subset,
        most_frequent_annotations,
        annotation_dict,
        output_name,
        zero_shot_classification,
        path,
        USA,
    )

    return regression_df


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.path
    output_name = args.output_name
    workers = args.workers
    annotation_threshold = args.annotation_threshold
    USA = args.USA

    pre_processing_pipeline(path, output_name, workers, annotation_threshold, USA)
