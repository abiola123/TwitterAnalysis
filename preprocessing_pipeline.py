import argparse
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
import re
from pyspark.conf import SparkConf


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
    default=750,
    help="Minimum number appearances for an annotation to be considered significant",
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

    most_frequent_annotations = annotations.filter(lambda x: x[1] > threshold).collect()

    # remove the first one which is 'Politics' (present in nearly all tweets)
    # TODO: check if this is also the case for celebrities
    most_frequent_annotations = list(map(lambda x: x[0], most_frequent_annotations))[1:]

    annotation_dict = {
        annotation: index for index, annotation in enumerate(most_frequent_annotations)
    }

    return most_frequent_annotations, annotation_dict


def extract_relevant_fields(rdd):
    return (
        rdd.filter(lambda x: x["data"])
        .flatMap(lambda x: x["data"])
        .filter(lambda x: x["entities"])
        .map(
            lambda x: {
                "tweet_text": x["text"],
                "tweet_date": x["created_at"],
                "tweet_hashtags": x["entities"]["hashtags"],
                "tweet_mentions": x["entities"]["mentions"],
                "tweet_urls": x["entities"]["urls"],
                "user_id": x["author_id"],
                "tweet_id": x["id"],
                "context_annotations": x["context_annotations"]
                if x["context_annotations"]
                else [],
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

    def get_period_of_day(x):
        hour = datetime.strptime(x["tweet_date"], "%Y-%m-%dT%H:%M:%S.%fZ").hour
        if hour >= 6 and hour < 12:
            return "morning"
        elif hour >= 12 and hour < 18:
            return "afternoon"
        else:
            return "night"

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

    def add_dummy_tweet_period(x):
        time_of_day = x["tweet_period"]

        x["dummy_tweet_period_morning"] = 0
        x["dummy_tweet_period_afternoon"] = 0
        x["dummy_tweet_period_night"] = 0

        x[f"dummy_tweet_period_{time_of_day}"] = 1

        return x

    # endregion

    # region PROCESSING PIPELINE
    # adding sentiment analysis on the tweet text using vader to the data
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
        lambda x: add_key_value(x, "tweet_period", get_period_of_day(x))
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

    json_rdd_data_fields = json_rdd_data_fields.map(lambda x: add_dummy_tweet_period(x))

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
            "tweet_period",
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


def pre_processing_pipeline(path, output_name, workers, annotation_threshold):
    spark = (
        SparkSession.builder.appName("tweet_loader")
        .master(f"local[{workers}]")
        .getOrCreate()
    )

    spark.sparkContext._conf.getAll()

    conf = spark.sparkContext._conf.setAll(
        [
            ("spark.executor.memory", "8g"),
            ("spark.driver.maxResultSize", "10g"),
            ("spark.driver.memory", "15g"),
        ]
    )

    spark.sparkContext.stop()

    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    print("**SparkContext created**")
    print(f"GUI: {spark.sparkContext.uiWebUrl}")
    print(f"AppName: {spark.sparkContext.appName}\n")

    rdd = load_data_set(spark, path)

    most_frequent_annotations, annotation_dict = get_most_frequent_annotations(
        rdd, annotation_threshold
    )

    rdd_subset = extract_relevant_fields(rdd)

    regression_df = apply_processing_pipeline(
        rdd, rdd_subset, most_frequent_annotations, annotation_dict, output_name
    )

    return regression_df


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.path
    output_name = args.output_name
    workers = args.workers
    annotation_threshold = args.annotation_threshold

    pre_processing_pipeline(path, output_name, workers, annotation_threshold)
