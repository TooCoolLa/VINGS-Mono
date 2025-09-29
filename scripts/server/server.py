import json
import websockets
import asyncio
import cv2, base64
import sys
import base64
import numpy as np
import csv
from PIL import Image
from PIL.ExifTags import TAGS
import io
import time
from datetime import datetime
import shutil
import os
class WebsocketServer:
    def __init__(self,host='0.0.0.0',port =5000,save = False,server2tracker_queue = None,mapper2server_queue=None):
        self.server2tracker_queue = server2tracker_queue
        self.save =save
        self.host =host
        self.port = port
        self.mapper2server_queue =mapper2server_queue
        
        self.yaw, self.pitch, self.raw, self.x, self.y, self.z, self.offsetx, self.offsety, self.offsetz = (
            1.5,
            0.3,
            3.1415926,
            0.4508,
            0.3173,
            0.3940,
            0.0,
            0.0,
            0.0,
        )
        self.frame = self.draw_corr()
        self.lock = asyncio.Lock()
        # self.evaluator = render.getEvaluator()
        if(self.save):
            folder_name=str(time.time())
            pic_save_dir =os.path.join("../mobile_data/",folder_name,'pic')
            self.save_dir=os.path.join("../mobile_data/",folder_name)
            if not os.path.exists(pic_save_dir):    
                os.mkdir(pic_save_dir)
            
    def delete_all_contents(self,folder_path):
        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            print(f"错误: {folder_path} 不是一个有效的文件夹路径")
            return
        
        # 遍历文件夹中的所有条目
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # 删除文件或符号链接
                    print(f"已删除文件: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹及其内容
                    print(f"已删除文件夹及其内容: {file_path}")
            except Exception as e:
                print(f"无法删除 {file_path}. 错误: {e}")
    
    def draw_corr(self):
        canvas = np.zeros((480, 720, 3), dtype="uint8")

        # 设置要显示的数字和其位置
        numbers = [
            "yaw" + str(self.yaw),
            "pitch" + str(self.pitch),
            "raw" + str(self.raw),
            "x" + str(self.x + self.offsetx),
            "y" + str(self.y + self.offsety),
            "z" + str(self.z + self.offsetz),
        ]
        positions = [(50, 50), (50, 100), (50, 150), (50, 200), (50, 250), (50, 300)]

        # 设置字体、颜色和大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # 白色
        thickness = 2

        # 将数字绘制到画面上
        for number, position in zip(numbers, positions):
            cv2.putText(canvas, number, position, font, font_scale, color, thickness)
        return canvas
    

    async def send(self,websocket):
        print("Client Connected!")
        await websocket.send("Connection Established")
        try:
            while True:
                # async with self.lock:
                    # C2W = render.Evaluator.RM_offset2C2W(
                    #     render.Evaluator.YawPitchRaw2RotationMatrix(self.yaw, self.pitch, self.raw),
                    #     [self.x + self.offsetx, self.y + self.offsety, self.z + self.offsetz],
                    # )
                # frame = self.evaluator.run(C2W, "")
                # cap = cv2.VideoCapture(0)
                # while cap.isOpened():
                #     _, frame = cap.read()
                while (self.mapper2server_queue.qsize()!=0):
                    self.frame = self.mapper2server_queue.get()
                encoded = cv2.imencode(".jpg", self.frame)[1]
                data = str(base64.b64encode(encoded))
                data = data[2 : len(data) - 1]
                await websocket.send(data)
                await asyncio.sleep(0.1) 
        except:
            print("Client Disconnected!")


    async def receive(self,websocket):
        gyroFlag = 0
        accFlag = 0
        if self.save:
            file_handle = open(os.path.join(self.save_dir,'imu.csv'), mode="w", newline="")
        gyroAccDict = {
            "timestamp": 0,
            "omega_x": 0,
            "omega_y": 0,
            "omega_z": 0,
            "alpha_x": 0,
            "alpha_y": 0,
            "alpha_z": 0,
        }
        if self.save:
            writer = csv.DictWriter(file_handle, fieldnames=gyroAccDict.keys())
            writer.writeheader()
        while True:
            message = await websocket.recv()  # Wait for a message from the client
            if (
                sys.getsizeof(message) < 2**10
            ):  # Todo: tell picture from string message in a more elegant way
                try:
                    message = json.loads(message)
                    if message["code"] == "orientationUpdate":
                        await self.orientationUpdate(message)
                    elif message["code"] == "scaleUpdate":
                        await self.scaleUpdate(message)
                    elif message["code"] == "scaleEnd":
                        await self.scaleEnd(message)
                    elif message["code"] == "accelerometer":
                        if gyroFlag == 1:
                            gyroAccDict["alpha_x"] = message["x"]
                            gyroAccDict["alpha_y"] = message["y"]
                            gyroAccDict["alpha_z"] = message["z"]
                        else:
                            gyroAccDict["alpha_x"] = message["x"]
                            gyroAccDict["alpha_y"] = message["y"]
                            gyroAccDict["alpha_z"] = message["z"]
                            gyroAccDict["timestamp"] = message["timestamp"] * 1000
                        accFlag = 1
                    elif message["code"] == "gyroscope":
                        if accFlag == 1:
                            gyroAccDict["omega_x"] = message["x"]
                            gyroAccDict["omega_y"] = message["y"]
                            gyroAccDict["omega_z"] = message["z"]
                        else:
                            gyroAccDict["omega_x"] = message["x"]
                            gyroAccDict["omega_y"] = message["y"]
                            gyroAccDict["omega_z"] = message["z"]
                            gyroAccDict["timestamp"] = message["timestamp"] * 1000
                        gyroFlag = 1
                    # print("Received message from client:", message)
                except:
                    pass
                if self.save:
                    if gyroFlag == 1 and accFlag == 1:
                        writer.writerow(gyroAccDict)
                        gyroFlag = 0
                        accFlag = 0
            else:
                message = json.loads(message)
                pic_base64 = message["image"]
                pic = base64.b64decode(pic_base64)
                image = Image.open(io.BytesIO(pic))
                exif_data = image._getexif()
                timestamp = 0
                datetime_original = None
                subsec_time_original = None
                subsec_seconds = None
                if exif_data:
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        if tag_name == "DateTimeOriginal":
                            datetime_original = value
                            print("datetime", datetime_original)
                        # 查找亚秒部分
                        elif tag_name == "SubsecTimeOriginal":
                            subsec_time_original = value
                            print("subsec", subsec_time_original)
                if datetime_original:
                    datetime_obj = datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S")

                    timestamp = int(time.mktime(datetime_obj.timetuple()))

                    if subsec_time_original:
                        subsec_seconds = str(subsec_time_original)
                if(self.server2tracker_queue):
                    if(self.server2tracker_queue.qsize()<2):
                        nparr = np.frombuffer(pic, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        self.server2tracker_queue.put({'rgb':img,'timestamp':str(timestamp)+subsec_seconds})
                        print("put data to server2tracker_queue")
                    else:
                        print("server2tracker_queue full")
                if self.save:         
                    with open(
                        os.path.join(self.save_dir,'pic', str(timestamp) + ".png"),
                        "wb",
                    ) as f:
                        f.write(pic)
                
    def YawPitchRaw2RotationMatrix(self,yaw, pitch, raw):
        Yaw = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
            dtype=np.float32,
        )
        Pitch = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        Raw = np.array(
            [[np.cos(raw), -np.sin(raw), 0], [np.sin(raw), np.cos(raw), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        return np.dot(np.dot(Yaw, Pitch), Raw)


    async def scaleUpdate(self,message):
        RotationMatrix = self.YawPitchRaw2RotationMatrix(self.yaw, self.pitch, self.raw)
        async with self.lock:
            self.offsetx = RotationMatrix[0][2] * (message["scale"] - 1)
            self.offsety = RotationMatrix[1][2] * (message["scale"] - 1)
            self.offsetz = RotationMatrix[2][2] * (message["scale"] - 1)


    async def orientationUpdate(self,message):

        async with self.lock:
            self.yaw += message["yaw"] / 200
            self.pitch -= message["pitch"] / 200


    async def scaleEnd(self,message):
        async with self.lock:
            self.x += self.offsetx
            self.y += self.offsety
            self.z += self.offsetz
            self.offsetx = 0
            self.offsety = 0
            self.offsetz = 0
            self.scale = 1
        print("yaw", self.yaw, "pitch", self.pitch, "raw", self.raw, "x", self.x, "y", self.y, "z", self.z)


    async def handle(self,websocket):
        send_task = asyncio.create_task(self.send(websocket))
        receive_task = asyncio.create_task(self.receive(websocket))
        await asyncio.gather(send_task, receive_task)


    async def run(self):
        async with websockets.serve(self.handle, self.host, self.port, max_size=2**30):
            print(f"WebSocket 服务器已启动，监听 ws://{self.host}:{self.port}")
            await asyncio.Future()


# usage
# server = WebsocketServer()
# asyncio.run(server.run())