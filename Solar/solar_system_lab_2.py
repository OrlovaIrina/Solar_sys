from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import math


class Formulas:
    @staticmethod
    def get_rot_mat(_axis, _angle_rad):
        if _axis == 'x':
            return np.array([[1, 0, 0],
                             [0, math.cos(_angle_rad), math.sin(_angle_rad)],
                             [0, -math.sin(_angle_rad), math.cos(_angle_rad)]])
        if _axis == 'y':
            return np.array([[math.cos(_angle_rad), 0, math.sin(_angle_rad)],
                             [0, 1, 0],
                             [-math.sin(_angle_rad), 0, math.cos(_angle_rad)]])
        if _axis == 'z':
            return np.array([[math.cos(_angle_rad), math.sin(_angle_rad), 0],
                             [-math.sin(_angle_rad), math.cos(_angle_rad), 0],
                             [0, 0, 1]])

        print("get_rot_mat: wrong input")
        exit(-9)

    @staticmethod
    def get_3d_rot_mat(_psi_rad: float = 0, _phi_rad: float = 0, _theta_rad: float = 0) -> np.ndarray:
        """
            Пересчет координат с учетом углов поворота орбиты
        """
        rot_z_1 = Formulas.get_rot_mat('z', _psi_rad)
        rot_x = Formulas.get_rot_mat('x', _theta_rad)
        rot_z_2 = Formulas.get_rot_mat('z', _phi_rad)

        rot_mat = rot_z_1
        rot_mat = rot_mat.dot(rot_x)
        rot_mat = rot_mat.dot(rot_z_2)
        return rot_mat

    @staticmethod
    def get_ellipse_pts(_orbit_a: float, _orbit_e: float) -> np.ndarray:
        assert _orbit_e < 1.0, "e: неверное!"

        b = _orbit_a*math.sqrt(1 - _orbit_e ** 2)
        pts = []
        angle_rng = np.linspace(0, 2*math.pi, 180, endpoint=True)
        for angle in angle_rng:
            x = _orbit_a * math.sin(angle)
            y = b * math.cos(angle)
            pts.append([x, y, 0])

        return np.array(pts)

    @staticmethod
    def get_ellipse_pts_shifted(_orbit_a: float, _orbit_e: float) -> np.ndarray:
        """ возвращает эллипс, у которого точка (0, 0, 0) находится в его фокусе"""
        assert _orbit_e < 1.0, "e: неверное!"

        b = _orbit_a * math.sqrt(1 - _orbit_e ** 2)
        pts = []
        angle_rng = np.linspace(0, 2 * math.pi, 180, endpoint=True)
        for angle in angle_rng:
            x = _orbit_a * (math.sin(angle) - _orbit_e)
            y = b * math.cos(angle)
            pts.append([x, y, 0])

        return np.array(pts)

    @staticmethod
    def satellite_pos_geostat_crd(_orbit_a, _orbit_e,  _t, _phi, _psi, _theta) -> np.ndarray:
        x = \
            - _orbit_a * math.sqrt(1 - _orbit_e ** 2) * math.sin(_t) \
            * (
                    math.cos(_psi) * math.cos(_theta) * math.sin(_phi) \
                    + math.cos(_phi) * math.sin(_psi)
            ) \
            + _orbit_a * (math.cos(_t) - _orbit_e) \
            * (
                    -    math.cos(_theta) * math.sin(_phi) * math.sin(_psi) \
                    + math.cos(_phi) * math.cos(_psi)
            )

        y = \
            _orbit_a * math.sqrt(1 - _orbit_e ** 2) * math.sin(_t) \
            * (
                    math.cos(_phi) * math.cos(_psi) * math.cos(_theta) \
                    - math.sin(_phi) * math.sin(_psi)
            ) \
            + _orbit_a * (math.cos(_t) - _orbit_e) \
            * (
                    math.cos(_phi) * math.cos(_theta) * math.sin(_psi) \
                    + math.cos(_psi) * math.sin(_phi)
            )

        z = \
            _orbit_a * math.sin(_theta) \
            * ( \
                        math.sqrt(1 - _orbit_e ** 2) * math.cos(_psi) * math.sin(_t) \
                        + (math.cos(_t) - _orbit_e) * math.sin(_psi) \
                )

        return np.array([[x, y, z]])


class CelestialObject:
    __R = None
    __list_of_plots = []
    __list_of_plots_pos = []

    def __init__(self, _R: float, _w: gl.GLViewWidget):
        self.__R = _R

        obj_meshdata = gl.MeshData.sphere(radius=_R, rows=50, cols=50)
        obj_meshitem = gl.GLMeshItem(meshdata=obj_meshdata, smooth=False, color=(0., 0.35, 0., 1.))
        obj_meshitem.setGLOptions('opaque')  # Непрозрачный
        self.__obj_data = obj_meshdata
        self.__obj_item = obj_meshitem
        _w.addItem(obj_meshitem)


    def change_color(self, c):
        self.__obj_item.setColor(c)

    #  ДОЛГОТЫ - зелено- желтая вертикальная сетка от Z к OXY
    def add_longitude(self):
        r = self.__R + 0.001
        phi_rng = np.linspace(0, 180, 6, endpoint=False)
        theta_rng = np.linspace(0, 360, 360, endpoint=True)
        cad = 0

        for phi in phi_rng:
            angle = (math.pi * phi) / 180
            phi_sin = math.sin(angle)
            phi_cos = math.cos(angle)

            i = 0
            # пересоздать массив - иначе будет только последний график
            pts = np.ndarray((theta_rng.size, 3), dtype=np.float32)

            for theta in theta_rng:
                angle = (math.pi * theta) / 180
                x = r * math.sin(angle) * phi_cos
                y = r * math.sin(angle) * phi_sin
                z = r * math.cos(angle)
                pts[i] = [x, y, z]
                i = i + 1

            cad = cad + (1. / phi_rng.size)
            plt = gl.GLLinePlotItem(pos=pts, color=(cad, 1., cad, 1.))
            plt.setGLOptions('opaque')
            self.__list_of_plots.append(plt)
            self.__list_of_plots_pos.append(plt.pos)
            w.addItem(plt)

    #  Широты - красная гориз сетка
    def add_latitude(self):
        r = self.__R + 0.001
        phi_rng = np.linspace(0., 360., 360, endpoint=True)
        theta_rng = np.linspace(10., 170., 10, endpoint=True)
        cad = 0

        for theta in theta_rng:
            angle = math.pi * (theta / 180)
            theta_sin = math.sin(angle)
            theta_cos = math.cos(angle)

            i = 0
            # пересоздать массив - иначе будет только последний график
            pts = np.ndarray(shape=(phi_rng.size, 3), dtype=np.float32)

            for phi in phi_rng:
                angle2 = (math.pi * phi) / 180
                x = r * math.cos(angle2) * theta_sin
                y = r * math.sin(angle2) * theta_sin
                z = r * theta_cos
                pts[i] = [x, y, z]
                i = i + 1

            cad = cad + int(255 / theta_rng.size)
            plt = gl.GLLinePlotItem(pos=pts, color=pg.glColor(250, cad, cad))
            plt.setGLOptions('opaque')
            self.__list_of_plots.append(plt)
            self.__list_of_plots_pos.append(plt.pos)
            w.addItem(plt)

    # ОСЬ вертикальная
    def add_axis(self):
        pts = np.array([[0., 0., -(self.__R + 0.5)], [0., 0., self.__R + 0.5]])
        plt = gl.GLLinePlotItem(pos=pts, color=pg.glColor(100, 10, 200), width=3)
        self.__list_of_plots_pos.append(plt.pos)
        self.__list_of_plots.append(plt)
        w.addItem(plt)

    def obj_data(self):
        return self.__obj_data

    def obj_item(self):
        return self.__obj_item

    def list_of_plots(self):
        return self.__list_of_plots

    def list_of_plots_pos(self):
        return self.__list_of_plots_pos


class Orbit:
    def __init__(self, _orbit_a: float, _orbit_e: float, _w: gl.GLViewWidget):
        self.__a = _orbit_a
        self.__e = _orbit_e
        self.__pts = Formulas.get_ellipse_pts_shifted(_orbit_a, _orbit_e)
        self.__psi = 0
        self.__theta = 0
        self.__phi = 0

        plt = gl.GLLinePlotItem(pos=self.__pts, color=(0, 1, 0, 1), width=2)
        self.__plt = plt
        _w.addItem(plt)

    def rotate(self, _psi_rad: float = 0, _theta_rad: float = 0, _phi_rad: float = 0):
        """повернуть график орбиты на углы Эйлера"""
        rot_mat = Formulas.get_3d_rot_mat(_psi_rad=_psi_rad, _theta_rad=_theta_rad, _phi_rad=_phi_rad)
        pts = np.dot(self.__pts, rot_mat)
        self.__plt.setData(pos=pts)

    def mrotate(self, _psi_rad: float = 0, _theta_rad: float = 0, _phi_rad: float = 0, _pts=[[0, 0, 0]]):
        """повернуть график орбиты на углы Эйлера"""
        rot_mat = Formulas.get_3d_rot_mat(_psi_rad=_psi_rad, _theta_rad=_theta_rad, _phi_rad=_phi_rad)
        pts = np.dot(self.__pts, rot_mat)
        pts = np.add(pts, _pts)
        self.__plt.setData(pos=pts, color=(0, 1, 1, 1))

    @property
    def a(self):
        return self.__a

    @property
    def e(self):
        return self.__e


class SolarSystem:
    def __init__(self):
        self.__3d_widget = SolarSystem.get_3d_widget()

    @staticmethod
    def get_3d_widget() -> gl.GLViewWidget:
        w = gl.GLViewWidget()
        w.opts['distance'] = 40
        w.showMaximized()
        w.setWindowTitle('pyqtgraph 3D example: GLLinePlotItem')

        # КООРДИНАТНЫЕ СЕТКИ
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        # w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        # w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        # w.addItem(gz)

        # AXIS
        size = QtGui.QVector3D(10, 10, 10)
        axis = gl.GLAxisItem(size, antialias=False)
        # z - green
        # y - yellow
        # x - blue
        w.addItem(axis)

        return w


def update():
    global orb, orbs, global_time_i, global_time_i_max, global_time_i_step
    global_time_i = (global_time_i + global_time_i_step) % global_time_i_max

    global global_phi, global_phi_max, global_psi, global_psi_max, global_theta, global_theta_max\
        , global_angle_degree_step

    is_changed = True
    if global_psi < global_psi_max:
        global_psi += global_angle_degree_step
    elif global_theta < global_theta_max:
        global_theta += global_angle_degree_step
    elif global_phi < global_phi_max:
        global_phi += global_angle_degree_step
    else:
        is_changed = False

    if is_changed:
        orb.rotate(
            _psi_rad=math.radians(global_psi)
            , _theta_rad=math.radians(global_theta)
            , _phi_rad=math.radians(global_phi))

    pts = Formulas.satellite_pos_geostat_crd(
        _orbit_a=orb.a
        , _orbit_e=orb.e
        , _psi=math.radians(global_psi)
        , _phi=math.radians(global_phi)
        , _theta=math.radians(global_theta)
        , _t=math.radians(global_time_i))
    # ------------------------------------------------

    # Шар Земли
    global Earth
    obj_data = Earth.obj_data()
    obj_item = Earth.obj_item()
    verts = obj_data.vertexes().copy()
    md = gl.MeshData(vertexes=verts, faces=obj_data.faces(), edges=obj_data.edges(),
                     vertexColors=obj_data.vertexColors(), faceColors=obj_data.faceColors())
    # 1 - Движение Земли по орбите
    verts = np.add(verts, pts)

    md.setVertexes(verts)
    obj_item.setMeshData(meshdata=md)

    # ======================
    # Вращение вокруг оси
    global axle_rotate_angle_z
    axle_rotate_angle_z = (axle_rotate_angle_z + (10 * math.pi / 180)) % (2 * math.pi)
    rot_mat = Formulas.get_3d_rot_mat(_psi_rad=math.radians(0)
                                        , _theta_rad=math.radians(0)
                                        , _phi_rad=axle_rotate_angle_z)
    # Наклон оси Земли
    axle_tilt = Formulas.get_3d_rot_mat(_psi_rad=math.radians(30)
                                      , _theta_rad=math.radians(30)
                                      , _phi_rad=math.radians(30))
    # Вращение вокруг оси + Наклон оси
    rot_mat = np.dot(rot_mat, axle_tilt)

    for plt1, plt1_pos in zip(Earth.list_of_plots(), Earth.list_of_plots_pos()):
        pos1 = plt1_pos.copy()
        # 1 - Вращение вокруг оси + Наклон Земли
        pos1 = np.dot(pos1, rot_mat)
        # 2 - Движение Земли по орбите
        pos1 = np.add(pos1, pts)

        plt1.setData(pos=pos1)
    #------------------------------------

    global moon
    ptsm = Formulas.satellite_pos_geostat_crd(
        _orbit_a=orbs.a
        , _orbit_e=orbs.e
        , _psi=math.radians(global_psi+30)
        , _phi=math.radians(global_phi+30)
        , _theta=math.radians(global_theta+30)
        , _t=math.radians(global_time_i*10))

    coor = np.add(pts, ptsm)
    moon.setData(pos=coor)

    orbs.mrotate(
        _psi_rad=math.radians(global_psi+30)
        , _theta_rad=math.radians(global_theta+30)
        , _phi_rad=math.radians(global_phi+30)
        , _pts=pts)


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseDesktopOpenGL)
    app = QtWidgets.QApplication(sys.argv)
    w = SolarSystem.get_3d_widget()

    axle_rotate_angle_z = 0

    Sun = CelestialObject(_R=4, _w=w)
    Sun.change_color((1., 1., 0., 1))

    Earth = CelestialObject(_R=2, _w=w)
    # Earth.change_color((1., 1., 0., 1))
    Earth.add_longitude()
    Earth.add_latitude()
    Earth.add_axis()

    # Орбита земли
    orb = Orbit(_orbit_a=15, _orbit_e=0.017, _w=w)

    # Луна
    moon = gl.GLScatterPlotItem(
        pos=np.array([[0, 0, 0]])
        , size=1
        , color=(0, 1, 1, 1)
        , pxMode=False)
    w.addItem(moon)

    # Орбита луны
    orbs = Orbit(_orbit_a=4, _orbit_e=0.017, _w=w)


    # ======================
    # ВРЕМЯ
    global_time_i = 0
    global_time_i_max = 3000
    global_time_i_step = 1

    global_phi = 0
    global_psi = 0
    global_theta = 0

    global_phi_max = 30
    global_psi_max = 50
    global_theta_max = 20
    global_angle_degree_step = 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)

    sys.exit(app.exec_())  # Запускаем цикл обработки событий
