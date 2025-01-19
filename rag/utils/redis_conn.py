import logging
import json
import os
import redis
from rag import settings
from rag.utils import singleton
from dotenv import load_dotenv

# Load environment variables from .env file (ensure you have the .env file set up with the required values)
load_dotenv()

# Redis connection helper function
def get_redis_client():
    """Create and return a Redis client."""
    redis_client = redis.StrictRedis(
        host=settings.REDIS["host"].split(":")[0],  # Assuming host is in 'hostname:port' format
        port=int(settings.REDIS["host"].split(":")[1]),
        db=int(settings.REDIS.get("db", 1)),  # Default to db 1
        username=os.getenv("REDIS_USER_NAME"),  # Fetch from .env or environment variables
        password=os.getenv("REDIS_PASSWORD"),  # Fetch from .env or environment variables
        decode_responses=True
    )
    try:
        # Test the connection
        if redis_client.ping():
            print("Successfully connected to Redis!")
        return redis_client
    except redis.AuthenticationError:
        logging.error("Authentication failed! Please check your username and password.")
        raise
    except Exception as e:
        logging.error(f"Failed to connect to Redis: {e}")
        raise


class Payload:
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        self.__message = json.loads(message["message"])

    def ack(self):
        try:
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            logging.warning(f"[EXCEPTION] ack {self.__queue_name} || {e}")
        return False

    def get_message(self):
        return self.__message


@singleton
class RedisDB:
    def __init__(self):
        self.REDIS = get_redis_client()  # Use the helper function to get the client
        self.config = settings.REDIS

    def health(self):
        try:
            self.REDIS.ping()
            a, b = "xx", "yy"
            self.REDIS.set(a, b, 3)
            if self.REDIS.get(a) == b:
                return True
        except Exception as e:
            logging.warning(f"Health check failed: {e}")
        return False

    def is_alive(self):
        return self.REDIS is not None

    def exist(self, k):
        if not self.REDIS:
            return
        try:
            return self.REDIS.exists(k)
        except Exception as e:
            logging.warning(f"RedisDB.exist {k} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed

    def get(self, k):
        if not self.REDIS:
            return
        try:
            return self.REDIS.get(k)
        except Exception as e:
            logging.warning(f"RedisDB.get {k} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed

    def set_obj(self, k, obj, exp=3600):
        try:
            self.REDIS.set(k, json.dumps(obj, ensure_ascii=False), exp)
            return True
        except Exception as e:
            logging.warning(f"RedisDB.set_obj {k} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return False

    def set(self, k, v, exp=3600):
        try:
            self.REDIS.set(k, v, exp)
            return True
        except Exception as e:
            logging.warning(f"RedisDB.set {k} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return False

    def sadd(self, key: str, member: str):
        try:
            self.REDIS.sadd(key, member)
            return True
        except Exception as e:
            logging.warning(f"RedisDB.sadd {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return False

    def srem(self, key: str, member: str):
        try:
            self.REDIS.srem(key, member)
            return True
        except Exception as e:
            logging.warning(f"RedisDB.srem {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return False

    def smembers(self, key: str):
        try:
            return self.REDIS.smembers(key)
        except Exception as e:
            logging.warning(f"RedisDB.smembers {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return None

    def zadd(self, key: str, member: str, score: float):
        try:
            self.REDIS.zadd(key, {member: score})
            return True
        except Exception as e:
            logging.warning(f"RedisDB.zadd {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return False

    def zcount(self, key: str, min: float, max: float):
        try:
            return self.REDIS.zcount(key, min, max)
        except Exception as e:
            logging.warning(f"RedisDB.zcount {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return 0

    def zpopmin(self, key: str, count: int):
        try:
            return self.REDIS.zpopmin(key, count)
        except Exception as e:
            logging.warning(f"RedisDB.zpopmin {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return None

    def zrangebyscore(self, key: str, min: float, max: float):
        try:
            return self.REDIS.zrangebyscore(key, min, max)
        except Exception as e:
            logging.warning(f"RedisDB.zrangebyscore {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return None

    def transaction(self, key, value, exp=3600):
        try:
            pipeline = self.REDIS.pipeline(transaction=True)
            pipeline.set(key, value, exp, nx=True)
            pipeline.execute()
            return True
        except Exception as e:
            logging.warning(f"RedisDB.transaction {key} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed
        return False

    def queue_product(self, queue, message, exp=settings.SVR_QUEUE_RETENTION) -> bool:
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                pipeline = self.REDIS.pipeline()
                pipeline.xadd(queue, payload)
                pipeline.execute()
                return True
            except Exception as e:
                logging.exception(f"RedisDB.queue_product {queue} got exception: {e}")
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> Payload:
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
            args = {
                "groupname": group_name,
                "consumername": consumer_name,
                "count": 1,
                "block": 10000,
                "streams": {queue_name: msg_id},
            }
            messages = self.REDIS.xreadgroup(**args)
            if not messages:
                return None
            stream, element_list = messages[0]
            msg_id, payload = element_list[0]
            return Payload(self.REDIS, queue_name, group_name, msg_id, payload)
        except Exception as e:
            if "key" in str(e):
                pass
            else:
                logging.exception(f"RedisDB.queue_consumer {queue_name} got exception: {e}")
        return None

    def get_unacked_for(self, consumer_name, queue_name, group_name):
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                return
            pendings = self.REDIS.xpending_range(
                queue_name,
                group_name,
                min=0,
                max=10000000000000,
                count=1,
                consumername=consumer_name,
            )
            if not pendings:
                return
            msg_id = pendings[0]["message_id"]
            msg = self.REDIS.xrange(queue_name, min=msg_id, count=1)
            _, payload = msg[0]
            return Payload(self.REDIS, queue_name, group_name, msg_id, payload)
        except Exception as e:
            if "key" in str(e):
                return
            logging.exception(f"RedisDB.get_unacked_for {consumer_name} got exception: {e}")
            self.REDIS = get_redis_client()  # Reconnect if needed

    def queue_info(self, queue, group_name) -> dict | None:
        try:
            groups = self.REDIS.xinfo_groups(queue)
            for group in groups:
                if group["name"] == group_name:
                    return group
        except Exception as e:
            logging.warning(f"RedisDB.queue_info {queue} got exception: {e}")
        return None


REDIS_CONN = RedisDB()
