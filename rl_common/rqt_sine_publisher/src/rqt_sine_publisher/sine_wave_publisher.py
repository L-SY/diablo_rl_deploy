#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from std_msgs.msg import Float64MultiArray

from python_qt_binding.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                         QLineEdit, QPushButton, QDoubleSpinBox, QScrollArea)
from python_qt_binding.QtCore import QTimer

from rqt_gui_py.plugin import Plugin


class ChannelWidget(QWidget):
    """
    单个通道的面板，包括幅度、频率、最小值、最大值和当前值输入。
    """
    def __init__(self, channel_id, parent=None):
        super(ChannelWidget, self).__init__(parent)
        self.channel_id = channel_id
        layout = QHBoxLayout()

        # 通道编号标签
        self.label = QLabel("通道 {}".format(channel_id))
        layout.addWidget(self.label)

        # 当前值输入
        layout.addWidget(QLabel("当前值"))
        self.current_spin = QDoubleSpinBox()
        self.current_spin.setRange(-1000.0, 1000.0)
        self.current_spin.setDecimals(3)
        self.current_spin.setSingleStep(0.1)
        self.current_spin.setValue(0.0)
        layout.addWidget(self.current_spin)

        # 幅度输入
        layout.addWidget(QLabel("幅度"))
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-1000.0, 1000.0)
        self.amp_spin.setDecimals(3)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.setValue(1.0)
        layout.addWidget(self.amp_spin)

        # 频率输入
        layout.addWidget(QLabel("频率"))
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.0, 1000.0)
        self.freq_spin.setDecimals(3)
        self.freq_spin.setSingleStep(0.1)
        self.freq_spin.setValue(1.0)
        layout.addWidget(self.freq_spin)

        # 最小值输入
        layout.addWidget(QLabel("最小值"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1000.0, 1000.0)
        self.min_spin.setDecimals(3)
        self.min_spin.setSingleStep(0.1)
        self.min_spin.setValue(-1.0)
        layout.addWidget(self.min_spin)

        # 最大值输入
        layout.addWidget(QLabel("最大值"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1000.0, 1000.0)
        self.max_spin.setDecimals(3)
        self.max_spin.setSingleStep(0.1)
        self.max_spin.setValue(1.0)
        layout.addWidget(self.max_spin)

        self.setLayout(layout)

    def get_params(self):
        """
        返回当前参数：幅度、频率、最小值、最大值
        """
        amp = self.amp_spin.value()
        freq = self.freq_spin.value()
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()
        return amp, freq, min_val, max_val

    def get_current_value(self):
        """
        返回当前值
        """
        return self.current_spin.value()

    def zero_all(self):
        """
        将所有参数置零
        """
        self.current_spin.setValue(0.0)
        self.amp_spin.setValue(0.0)
        self.freq_spin.setValue(0.0)
        self.min_spin.setValue(0.0)
        self.max_spin.setValue(0.0)


class RqtSinePublisher(Plugin):
    """
    RQT 插件：根据用户设置的各通道参数计算正弦波数据，
    或发布用户设定的单帧数据到指定话题。
    """
    def __init__(self, context):
        super(RqtSinePublisher, self).__init__(context)
        self.setObjectName('rqt_sine_publisher')

        # 创建主界面
        self._widget = QWidget()
        self._widget.setWindowTitle('rqt_sine_publisher')

        main_layout = QVBoxLayout()

        # 1. 话题名称输入区域
        topic_layout = QHBoxLayout()
        topic_layout.addWidget(QLabel("话题名称:"))
        self.topic_line_edit = QLineEdit()
        self.topic_line_edit.setPlaceholderText("输入话题名称")
        self.topic_line_edit.setText("/sine_wave_values")  # 默认话题名称
        topic_layout.addWidget(self.topic_line_edit)
        self.update_topic_button = QPushButton("更新话题")
        topic_layout.addWidget(self.update_topic_button)
        self.update_topic_button.clicked.connect(self.update_topic)
        main_layout.addLayout(topic_layout)

        # 2. 输入通道数量区域
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("通道数量:"))
        self.count_line_edit = QLineEdit()
        self.count_line_edit.setPlaceholderText("输入通道数量")
        count_layout.addWidget(self.count_line_edit)
        self.create_channels_button = QPushButton("生成通道")
        count_layout.addWidget(self.create_channels_button)
        self.create_channels_button.clicked.connect(self.generate_channels)
        main_layout.addLayout(count_layout)

        # 3. 中间区域：滚动区域用于显示多个通道面板
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout()
        self.channels_container.setLayout(self.channels_layout)
        self.scroll_area.setWidget(self.channels_container)
        main_layout.addWidget(self.scroll_area)

        # 4. 底部按钮区域：全部置零、开启发送、停止发送、发布单帧
        btn_layout = QHBoxLayout()

        self.zero_button = QPushButton("全部置零")
        self.zero_button.clicked.connect(self.zero_all)
        btn_layout.addWidget(self.zero_button)

        self.publish_single_button = QPushButton("发布单帧")
        self.publish_single_button.clicked.connect(self.publish_single_frame)
        btn_layout.addWidget(self.publish_single_button)

        self.start_button = QPushButton("开启发送")
        self.start_button.clicked.connect(self.start_publishing)
        btn_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止发送")
        self.stop_button.clicked.connect(self.stop_publishing)
        btn_layout.addWidget(self.stop_button)

        main_layout.addLayout(btn_layout)

        self._widget.setLayout(main_layout)
        context.add_widget(self._widget)

        # 保存生成的通道面板
        self.channel_widgets = []

        # 定时器：每 100 毫秒计算并发布正弦波数据（10Hz）
        self.timer = QTimer(self._widget)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.publish_values)

        # 初始化 ROS 消息发布器
        self.pub = None
        self.create_publisher(self.topic_line_edit.text())
        self.start_time = None

    def create_publisher(self, topic_name):
        """
        创建新的发布器
        """
        if self.pub is not None:
            self.pub.unregister()
        self.pub = rospy.Publisher(topic_name, Float64MultiArray, queue_size=10)
        rospy.loginfo(f"Created publisher for topic: {topic_name}")

    def update_topic(self):
        """
        更新发布话题
        """
        # 如果正在发布，先停止
        if self.timer.isActive():
            self.stop_publishing()

        # 更新发布器
        new_topic = self.topic_line_edit.text()
        self.create_publisher(new_topic)
        rospy.loginfo(f"Updated topic to: {new_topic}")

    def generate_channels(self):
        """
        根据输入的通道数量生成对应的通道参数面板，
        首先清除已生成的面板。
        """
        # 清空已有面板
        for i in reversed(range(self.channels_layout.count())):
            widget_to_remove = self.channels_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)
        self.channel_widgets = []

        try:
            count = int(self.count_line_edit.text())
        except ValueError:
            rospy.logerr("请输入有效的通道数量")
            return

        # 生成相应数量的 ChannelWidget
        for i in range(count):
            channel_widget = ChannelWidget(i + 1)
            self.channels_layout.addWidget(channel_widget)
            self.channel_widgets.append(channel_widget)

    def zero_all(self):
        """
        一键将所有通道面板中的参数置零
        """
        for widget in self.channel_widgets:
            widget.zero_all()

    def publish_single_frame(self):
        """
        发布单帧数据，使用每个通道的当前值
        """
        msg = Float64MultiArray()
        data = []
        for widget in self.channel_widgets:
            value = widget.get_current_value()
            data.append(value)
        msg.data = data
        self.pub.publish(msg)
        rospy.loginfo(f"Published single frame: {data}")

    def start_publishing(self):
        """
        开启定时器，开始周期性发布消息
        """
        if not self.timer.isActive():
            self.start_time = rospy.get_time()
            self.timer.start()

    def stop_publishing(self):
        """
        停止定时器，停止消息发布
        """
        if self.timer.isActive():
            self.timer.stop()

    def publish_values(self):
        """
        定时器回调函数：计算每个通道的正弦波值，
        并对数值进行限幅处理后发布消息。
        """
        current_time = rospy.get_time()
        dt = current_time - self.start_time if self.start_time is not None else 0.0
        msg = Float64MultiArray()
        data = []
        for widget in self.channel_widgets:
            amp, freq, min_val, max_val = widget.get_params()
            # 计算正弦波数值：amp * sin(2π * freq * dt)
            value = amp * math.sin(2 * math.pi * freq * dt)
            # 限幅操作
            if value < min_val:
                value = min_val
            if value > max_val:
                value = max_val
            data.append(value)
        msg.data = data
        self.pub.publish(msg)

    def shutdown_plugin(self):
        """
        插件关闭时停止定时器和发布器
        """
        self.stop_publishing()
        if self.pub is not None:
            self.pub.unregister()

    def save_settings(self, plugin_settings, instance_settings):
        # 可保存插件状态
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # 可恢复插件状态
        pass

if __name__ == '__main__':
    # 用于独立测试插件界面
    import sys
    from python_qt_binding.QtWidgets import QApplication
    rospy.init_node('rqt_sine_publisher_rqt_test')
    app = QApplication(sys.argv)
    plugin = RqtSinePublisher(context=None)
    plugin._widget.show()
    sys.exit(app.exec_())