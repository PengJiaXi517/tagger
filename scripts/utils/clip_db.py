import json
import logging

import pandas as pd
import pymysql
from dbutils.pooled_db import PooledDB


class ClipDB:

    def __init__(self, max_connection=2):
        self.pool = PooledDB(
            creator=pymysql,
            host="10.199.2.234",
            user="BagSlice",
            password="Sti@123456",
            database="bag_slice",
            port=3306,
            autocommit=False,  # 如果需要修改数据库，这里可以改成 True
            maxconnections=max_connection,
        )
        logging.basicConfig(level=logging.INFO)

    def get_clips_by_gps(self, lon=121.515147, lat=31.23716, distance=0.0001):
        """查询轨迹上点到给定 GPS 最小值小于 distance 的 clip 列表
        lon: 经度
        lat: 纬度
        distance: 距离（单位°，深圳地区约等于 1e5 m)
        """
        conn = self.pool.connection()
        try:
            query = f"""
                SELECT clip_name, task_id
                FROM clip_geo_data
                WHERE ST_DISTANCE(
                    positions, 
                    ST_GeomFromText('POINT({lon} {lat})')
                ) <= {distance} AND task_id is not null
            """
            logging.info(f"Executing query: {query}")
            df = pd.read_sql(query, conn)
            return df.to_dict(orient="records")
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return []
        finally:
            conn.close()

    def get_clips_by_polygon(self, geojson_data):
        """查询轨迹处于给定 polygon 内 的 clip 列表
        geojson_data: 一个只包含一个 polygon feature 的 GeoJSON, 可以通过 https://geojson.io/ 手动绘制得到
        """
        conn = self.pool.connection()
        try:
            polygon_coords = geojson_data["features"][0]["geometry"]["coordinates"][0]
            polygon_wkt = (
                "POLYGON(("
                + ", ".join([f"{lon} {lat}" for lon, lat in polygon_coords])
                + "))"
            )
            query = f"""
                SELECT clip_name, task_id
                FROM clip_geo_data
                WHERE ST_Intersects(
                    positions,
                    ST_GeomFromText('{polygon_wkt}')
                ) AND task_id is not null
            """
            logging.info(f"Executing query: {query}")
            df = pd.read_sql(query, conn)
            return df.to_dict(orient="records")
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return []
        finally:
            conn.close()

    def close_pool(self):
        """关闭连接池"""
        self.pool.close()
        logging.info("Connection pool closed.")


if __name__ == "__main__":
    db = ClipDB()

    # gps_clips = db.get_clips_by_gps(
    #     lon=113.969309824, lat=22.585517670, distance=0.0002
    # )
    gps_clips = db.get_clips_by_gps(
        lon=121.58974862125837, lat=31.218359233882435, distance=0.0001
    )
    # print(json.dumps(gps_clips, indent=2, ensure_ascii=False))

    with open("./test_jsons/zhuangzhuzi_test.json", "w") as f:
        json.dump(gps_clips, f, indent=2)

    db.close_pool()

    # geojson_data = {
    #     "type": "FeatureCollection",
    #     "features": [
    #         {
    #             "type": "Feature",
    #             "properties": {},
    #             "geometry": {
    #                 "coordinates": [
    #                     [
    #                         [113.96028412829241, 22.584705334173265],
    #                         [113.96034921328192, 22.582416742815667],
    #                         [113.96333769912627, 22.582266505729834],
    #                         [113.96327261413433, 22.58524617734146],
    #                         [113.96206854180008, 22.585151029160755],
    #                         [113.96028412829241, 22.584705334173265],
    #                     ]
    #                 ],
    #                 "type": "Polygon",
    #             },
    #         }
    #     ],
    # }

    # polygon_clips = db.get_clips_by_polygon(geojson_data)
    # print(json.dumps(polygon_clips, indent=2, ensure_ascii=False))

    # db.close_pool()
