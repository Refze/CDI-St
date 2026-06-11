"""bcdi_gui.py v7.0 — dislocation loops, CIF import, rect detector, presets, export, measure, NN reconstruction."""
import sys,tempfile,os
import numpy as np
from collections import Counter
from PyQt6.QtCore import Qt,QThread,pyqtSignal,QUrl,QSize,QTimer,QRect
from PyQt6.QtGui import QPalette,QColor,QFont,QPixmap,QPainter,QFontMetrics,QBrush,QPen
from PyQt6.QtWidgets import (QApplication,QMainWindow,QWidget,QTabWidget,QVBoxLayout,QHBoxLayout,
    QFormLayout,QGridLayout,QGroupBox,QLabel,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,
    QLineEdit,QTableWidget,QTableWidgetItem,QHeaderView,QProgressBar,QMessageBox,QStatusBar,
    QSplitter,QPlainTextEdit,QFrame,QCheckBox,QScrollArea,QFileDialog,QSplashScreen)
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView; _HAS_WEB=True
except: _HAS_WEB=False
import matplotlib; matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .bcdi_core import (MATERIAL_PRESETS,DETECTOR_PRESETS,BCDIConfig,CrystalBuilder,ReflectionCalculator,
    BCDISimulator,DislocationConfig,DislocationLoopConfig,lattice_html_with_controls,bragg_html_with_controls,
    add_experimental_noise,export_measurement,load_cif_as_preset,find_preset,default_shape_for_material,compatible_shapes)
try:
    from cdi_st.nn_gui_tabs import T_Gen,T4,T4_Sup,T5,T6;_HAS_NN=True
except ImportError: _HAS_NN=False

QSS="""QMainWindow,QWidget{background:#0d1117;color:#e6edf3;font-family:'Segoe UI','Consolas',sans-serif;font-size:10pt}
QTabWidget::pane{border:1px solid #30363d;background:#0d1117;top:-1px}QTabBar::tab{background:#161b22;color:#8b949e;padding:10px 22px;margin-right:2px;border:1px solid #30363d;border-bottom:none;border-top-left-radius:6px;border-top-right-radius:6px;font-weight:600;min-width:140px}QTabBar::tab:selected{background:#0d1117;color:#4f98a3;border-bottom:2px solid #4f98a3}QTabBar::tab:disabled{color:#30363d}
QGroupBox{border:1px solid #30363d;border-radius:6px;margin-top:12px;padding-top:12px;color:#4f98a3;font-weight:700;background:#0f141b}QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 8px;left:10px}
QPushButton{background:#238636;color:#fff;border:none;padding:8px 18px;border-radius:5px;font-weight:600;min-height:26px}QPushButton:hover{background:#2ea043}QPushButton:disabled{background:#30363d;color:#6e7681}
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:4px 7px;color:#e6edf3}QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus{border:1px solid #4f98a3}
QComboBox::drop-down{border:none;width:20px}QComboBox QAbstractItemView{background:#161b22;border:1px solid #30363d;selection-background-color:#1f6feb;color:#e6edf3}
QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:4px;text-align:center;color:#e6edf3;height:16px}QProgressBar::chunk{background:#4f98a3;border-radius:3px}
QStatusBar{background:#161b22;color:#8b949e;border-top:1px solid #30363d}
QTableWidget{background:#0d1117;gridline-color:#30363d;border:1px solid #30363d;selection-background-color:#1f6feb;alternate-background-color:#161b22}QHeaderView::section{background:#161b22;color:#4f98a3;padding:5px;border:none;border-right:1px solid #30363d;border-bottom:1px solid #30363d;font-weight:700}
QPlainTextEdit{font-family:'Consolas',monospace;font-size:9pt;color:#c0c4cc;background:#010409}QCheckBox{color:#e6edf3;spacing:6px}QScrollArea{border:none;background:transparent}"""
MPL_DARK={'figure.facecolor':'#0d1117','axes.facecolor':'#000005','axes.edgecolor':'#30363d','axes.labelcolor':'#e6edf3','axes.titlecolor':'#e6edf3','xtick.color':'#8b949e','ytick.color':'#8b949e','text.color':'#e6edf3','font.size':9}
def _ph(m): return f'<html><head><style>html,body{{height:100%;margin:0;background:#0a0d12;color:#8b949e}}.w{{height:100%;display:flex;align-items:center;justify-content:center}}.i{{padding:24px;border:1px dashed #30363d;border-radius:8px}}</style></head><body><div class="w"><div class="i">{m}</div></div></body></html>'
def _dbl(lo,hi,v,d,suf=""):
    s=QDoubleSpinBox();s.setRange(lo,hi);s.setDecimals(d);s.setValue(v);s.setMinimumWidth(100)
    if suf: s.setSuffix(suf)
    return s
def _nb():
    b=QPushButton("next >>");b.setEnabled(False);b.setFixedSize(QSize(80,28));b.setStyleSheet("background:#238636;font-size:9pt;padding:4px 10px;min-height:20px;font-weight:700");return b

class BW(QThread):
    progress=pyqtSignal(int);log=pyqtSignal(str);done=pyqtSignal(object);failed=pyqtSignal(str)
    def __init__(self,c): super().__init__();self.c=c
    def run(self):
        try:
            self.log.emit(f"Building {self.c.MATERIAL_NAME}...");b=CrystalBuilder(self.c);b.build(lambda p:self.progress.emit(p));b.apply_shape_filter()
            if self.c.DISLOCATION: self.log.emit(f"Line dislocation {self.c.DISLOCATION.dtype}");b.apply_dislocation_displacement()
            if self.c.DISLOCATION_LOOP: self.log.emit(f"Dislocation loop R={self.c.DISLOCATION_LOOP.radius_angstrom:.0f}A");b.apply_dislocation_loop()
            n=len(b.supercell_positions_ang);sz=self.c.particle_size_nm;self.log.emit(f"Done: {n:,} atoms {sz[0]:.1f}x{sz[1]:.1f}x{sz[2]:.1f}nm");self.done.emit(b)
        except Exception as e: import traceback;self.failed.emit(f"{e}\n{traceback.format_exc()}")
class SW(QThread):
    progress=pyqtSignal(int);log=pyqtSignal(str);done=pyqtSignal(object,object,object);failed=pyqtSignal(str)
    def __init__(self,c,b,h): super().__init__();self.c=c;self.b=b;self.h=h
    def run(self):
        try:
            self.log.emit("Reflections...");self.progress.emit(5);rc=ReflectionCalculator(self.c,self.b);df=rc.calculate();self.log.emit(f"{len(df)} reflections");self.progress.emit(20)
            refl=rc.select_reflection(self.h);self.log.emit(f"{refl['hkl_str']} 2th={np.degrees(refl['two_theta_rad']):.3f}");self.progress.emit(30)
            sim=BCDISimulator(self.c,refl,self.b);sim.simulate(lambda p:self.progress.emit(30+int(.7*p)));self.log.emit(f"max={sim.diff_volume.max():.2e}");self.progress.emit(100);self.done.emit(sim,refl,df)
        except Exception as e: import traceback;self.failed.emit(f"{e}\n{traceback.format_exc()}")

# ======================= TAB 1 =======================
class T1(QWidget):
    built=pyqtSignal(object,object)
    def __init__(self): super().__init__();self.builder=None;self.config=None;self._w=None;self._ui()
    def _ui(self):
        root=QHBoxLayout(self);root.setContentsMargins(10,10,10,10);root.setSpacing(10)
        sc=QScrollArea();sc.setWidgetResizable(True);sc.setFixedWidth(520);inner=QWidget();ll=QVBoxLayout(inner);ll.setSpacing(6);ll.setContentsMargins(4,4,8,4)
        h=QHBoxLayout();t=QLabel("Material & Lattice");t.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700");h.addWidget(t,1);self.nb=_nb();h.addWidget(self.nb);ll.addLayout(h)
        # Material with CIF import
        mb=QGroupBox("Material");mv=QVBoxLayout(mb);mv.setSpacing(4)
        r1=QHBoxLayout();r1.addWidget(QLabel("Preset:"));self.combo=QComboBox()
        self.combo.setToolTip(
            "Pre-configured materials. Each preset contains the lattice "
            "constants (a, b, c, angles), space group, atomic basis, and "
            "common BCDI reflections.\n"
            "Includes: Si, Ge, Pt, Au, Cu, Pd, GaAs, GaN, SiC, ZnO,\n"
            "BaTiO3, SrTiO3, Fe5GeTe2, V2O3, hexagonal/cubic polytypes,\n"
            "and any custom .cif files you've loaded."
        )
        for k in MATERIAL_PRESETS: self.combo.addItem(k)
        self.combo.currentTextChanged.connect(self._pc);r1.addWidget(self.combo,1);mv.addLayout(r1)
        r2=QHBoxLayout();r2.addWidget(QLabel("Formula:"));self.fe=QLineEdit();self.fe.setPlaceholderText("e.g. Fe5GeTe2")
        self.fe.setToolTip(
            "Type a chemical formula (e.g. 'Si', 'GaAs', 'Fe5GeTe2'),\n"
            "then click 'Match' to find the matching preset. Useful when\n"
            "you know the formula but not the exact preset name."
        )
        r2.addWidget(self.fe,1)
        mb2=QPushButton("Match");mb2.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px");mb2.setMaximumWidth(55)
        mb2.setToolTip("Find the preset that matches the typed formula.")
        mb2.clicked.connect(self._mt);r2.addWidget(mb2)
        cif_btn=QPushButton("Load .cif");cif_btn.setStyleSheet("background:#1f6feb;padding:5px 10px;min-height:20px");cif_btn.setMaximumWidth(70)
        cif_btn.setToolTip(
            "Import a Crystallographic Information File (.cif) as a custom\n"
            "material. The file is parsed for cell parameters, space group,\n"
            "and atomic basis. The new preset becomes selectable in the dropdown."
        )
        cif_btn.clicked.connect(self._load_cif);r2.addWidget(cif_btn)
        mv.addLayout(r2);self.il=QLabel();self.il.setWordWrap(True);self.il.setStyleSheet("color:#8b949e;font-size:9pt");mv.addWidget(self.il);ll.addWidget(mb)
        # Supercell
        sb=QGroupBox("Supercell");sf=QFormLayout(sb);sf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);sf.setVerticalSpacing(4)
        row=QHBoxLayout();self.nx=QSpinBox();self.ny=QSpinBox();self.nz=QSpinBox()
        for s in(self.nx,self.ny,self.nz): s.setRange(1,2500);s.setValue(20);s.setMinimumWidth(55);s.valueChanged.connect(self._us)
        self.nx.setToolTip(
            "Number of unit cells along X axis. Sets the X dimension of\n"
            "the particle: physical size = nx × a (lattice constant)."
        )
        self.ny.setToolTip(
            "Number of unit cells along Y axis. Sets the Y dimension of\n"
            "the particle: physical size = ny × b."
        )
        self.nz.setToolTip(
            "Number of unit cells along Z axis. Sets the Z dimension of\n"
            "the particle: physical size = nz × c."
        )
        row.addWidget(QLabel("nx"));row.addWidget(self.nx);row.addWidget(QLabel("ny"));row.addWidget(self.ny);row.addWidget(QLabel("nz"));row.addWidget(self.nz);row.addStretch()
        sf.addRow("Cells:",row);self.shp=QComboBox()
        self.shp.setToolTip(
            "Particle shape carved from the rectangular supercell.\n"
            "  • cube:        the whole box\n"
            "  • sphere:       inscribed sphere\n"
            "  • cylinder:     inscribed cylinder along Z\n"
            "  • hexagonal_prism:  hexagonal cross-section\n"
            "  • octahedron / dodecahedron: faceted shapes\n"
            "Available shapes depend on the material's symmetry."
        )
        sf.addRow("Shape:",self.shp);self.sl=QLabel();self.sl.setStyleSheet("color:#8b949e;font-size:9pt");sf.addRow(self.sl);ll.addWidget(sb)
        vr=QHBoxLayout();self.bc=QCheckBox("Bonds");self.bc.setToolTip("Draw atomic bonds (slows rendering with many atoms)")
        vr.addWidget(self.bc);self.stc=QCheckBox("Strain");self.stc.setToolTip("Color atoms by local strain magnitude (if strain is enabled)")
        vr.addWidget(self.stc);self.dvc=QCheckBox("Disloc. viz");self.dvc.setToolTip("Highlight atoms near the dislocation core")
        vr.addWidget(self.dvc);vr.addStretch();ll.addLayout(vr)
        # Dislocation: line + loop tabs
        db=QGroupBox("Defects");dv=QVBoxLayout(db);dv.setSpacing(4)
        # Line dislocation
        self.dck=QCheckBox("Line dislocation");self.dck.setToolTip(
            "Add a single straight-line dislocation. The displacement field\n"
            "follows Volterra elasticity (edge: Burgers vector ⊥ line, screw:\n"
            "Burgers vector ∥ line, mixed: arbitrary angle)."
        );dv.addWidget(self.dck)
        self.df=QFrame();dg=QGridLayout(self.df);dg.setContentsMargins(2,2,2,2);dg.setHorizontalSpacing(8);dg.setVerticalSpacing(5)
        self.dt=QComboBox();self.dt.addItems(["edge","screw","mixed"])
        self.dt.setToolTip(
            "Dislocation character:\n"
            "  • edge:   Burgers vector perpendicular to line\n"
            "  • screw:  Burgers vector parallel to line\n"
            "  • mixed:  combination of both"
        )
        self.dd=QComboBox();self.dd.addItems(["Z","Y","X"])
        self.dd.setToolTip("Direction of the dislocation line in crystal axes.")
        dg.addWidget(QLabel("Type:"),0,0,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dt,0,1);dg.addWidget(QLabel("Line:"),0,2,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dd,0,3)
        self.dpx=_dbl(0,1,.5,3);self.dpx.setToolTip("X position of the dislocation core as a fraction (0-1) of the particle.")
        self.dpy=_dbl(0,1,.5,3);self.dpy.setToolTip("Y position of the dislocation core as a fraction (0-1) of the particle.")
        dg.addWidget(QLabel("X:"),1,0,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dpx,1,1);dg.addWidget(QLabel("Y:"),1,2,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dpy,1,3)
        self.db_=_dbl(0,50,0,3," A");self.db_.setSpecialValueText("auto")
        self.db_.setToolTip(
            "Magnitude of the Burgers vector in Ångström.\n"
            "Set to 0 (= 'auto') to use the lattice constant 'a'."
        )
        self.dnu=_dbl(0,.5,.3,3);self.dnu.setToolTip(
            "Poisson's ratio ν. Controls the radial vs. tangential strain ratio\n"
            "for edge/mixed dislocations. Typical metal: 0.3."
        )
        dg.addWidget(QLabel("|b|:"),2,0,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.db_,2,1);dg.addWidget(QLabel("nu:"),2,2,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dnu,2,3)
        self.df.setVisible(False);dv.addWidget(self.df);self.dck.toggled.connect(self.df.setVisible)
        # Loop dislocation
        self.lck=QCheckBox("Dislocation loop (prismatic)")
        self.lck.setToolTip(
            "Add a prismatic dislocation loop (closed circular line dislocation).\n"
            "Common defect in irradiated and quenched materials."
        )
        dv.addWidget(self.lck)
        self.lf=QFrame();lg=QGridLayout(self.lf);lg.setContentsMargins(2,2,2,2);lg.setHorizontalSpacing(8);lg.setVerticalSpacing(5)
        self.lcx=_dbl(0,1,.5,3);self.lcx.setToolTip("X position of the loop center (fraction 0-1).")
        self.lcy=_dbl(0,1,.5,3);self.lcy.setToolTip("Y position of the loop center (fraction 0-1).")
        self.lcz=_dbl(0,1,.5,3);self.lcz.setToolTip("Z position of the loop center (fraction 0-1).")
        lg.addWidget(QLabel("Cx:"),0,0,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lcx,0,1);lg.addWidget(QLabel("Cy:"),0,2,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lcy,0,3);lg.addWidget(QLabel("Cz:"),0,4,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lcz,0,5)
        self.lr=_dbl(1,500,20,1," A");self.lr.setToolTip("Loop radius in Ångström.")
        self.lb=_dbl(0,50,0,3," A");self.lb.setSpecialValueText("auto")
        self.lb.setToolTip("Burgers vector magnitude in Ångström (= 'auto' uses 'a').")
        self.ln=QComboBox();self.ln.addItems(["Z","Y","X"])
        self.ln.setToolTip("Loop normal direction (the loop plane is perpendicular to this).")
        self.lnu2=_dbl(0,.5,.3,3);self.lnu2.setToolTip("Poisson's ratio ν (typical metal: 0.3).")
        lg.addWidget(QLabel("R:"),1,0,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lr,1,1);lg.addWidget(QLabel("|b|:"),1,2,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lb,1,3);lg.addWidget(QLabel("N:"),1,4,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.ln,1,5)
        self.lf.setVisible(False);dv.addWidget(self.lf);self.lck.toggled.connect(self.lf.setVisible)
        ll.addWidget(db)
        stb=QGroupBox("Analytical strain");stf=QFormLayout(stb);stf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);stf.setVerticalSpacing(4)
        self.strc=QComboBox();self.strc.addItems(["none","radial_gradient","edge_dislocation","random"])
        self.strc.setToolTip(
            "Add a parameterized strain field on top of the rigid lattice:\n"
            "  • none:              no strain (default)\n"
            "  • radial_gradient:   strain grows from center outward\n"
            "  • edge_dislocation:  Volterra edge-dislocation field\n"
            "  • random:            Gaussian random field (smoothed)"
        )
        stf.addRow("Type:",self.strc)
        self.strm=_dbl(0,.1,1e-4,6);self.strm.setToolTip(
            "Peak strain magnitude (dimensionless). Typical BCDI: 1e-4 to 1e-3.\n"
            "0 = no strain."
        )
        stf.addRow("eps:",self.strm);ll.addWidget(stb)
        self.bb=QPushButton("Build & Visualize");self.bb.setStyleSheet("background:#1f6feb;min-height:34px;font-size:11pt")
        self.bb.setToolTip("Build the supercell, apply strain/dislocations, and render the 3D lattice.")
        self.bb.clicked.connect(self._bd);ll.addWidget(self.bb)
        self.pg=QProgressBar();ll.addWidget(self.pg);self.lg=QPlainTextEdit();self.lg.setReadOnly(True);self.lg.setMaximumHeight(65);ll.addWidget(self.lg);ll.addStretch();sc.setWidget(inner);root.addWidget(sc)
        right=QWidget();rv=QVBoxLayout(right);rv.setContentsMargins(0,0,0,0);rv.addWidget(QLabel("3D Lattice"))
        if _HAS_WEB: self.viz=QWebEngineView();self.viz.setHtml(_ph("Build lattice."))
        else: self.viz=QLabel("Need PyQt6-WebEngine")
        self.viz.setMinimumHeight(500);rv.addWidget(self.viz,1);root.addWidget(right,1);self._pc(self.combo.currentText());self._us()
    def _load_cif(self):
        path,_=QFileDialog.getOpenFileName(self,"Load CIF","","CIF files (*.cif);;All (*)")
        if not path: return
        try:
            mat=load_cif_as_preset(path);name=mat['formula']
            if name in MATERIAL_PRESETS: name+="_cif"
            MATERIAL_PRESETS[name]=mat;self.combo.addItem(name);self.combo.setCurrentText(name)
            self.lg.appendPlainText(f"Loaded {name}: {len(mat['basis'])} atoms, {mat['space_group']}")
        except Exception as e: QMessageBox.critical(self,"CIF Error",str(e))
    def _pc(self,n):
        if n not in MATERIAL_PRESETS: return
        m=MATERIAL_PRESETS[n];a=m['a'];b=m.get('b') or a;c=m.get('c') or a
        cnt=Counter(m['species_per_site']);spc=', '.join(f'{k}:{v}' for k,v in cnt.items())
        self.il.setText(f"<b>{m.get('formula',n)}</b> {m['space_group']}<br>a={a:.3f} b={b:.3f} c={c:.3f} g={m['gamma']}<br>{len(m['basis'])} atoms/cell: {spc}")
        shapes=compatible_shapes(n);ds=default_shape_for_material(n);self.shp.blockSignals(True);self.shp.clear();self.shp.addItems(shapes)
        if ds in shapes: self.shp.setCurrentText(ds)
        self.shp.blockSignals(False);self._us()
    def _mt(self):
        q=self.fe.text().strip()
        if q:
            r=find_preset(q)
            if r: self.combo.setCurrentText(r)
            else: QMessageBox.information(self,"No match",f"No preset for '{q}'. Try Load .cif")
    def _us(self):
        try:
            cfg=BCDIConfig(self.combo.currentText());cfg.SUPERCELL_MULT=(self.nx.value(),self.ny.value(),self.nz.value())
            sz=cfg.particle_size_nm;nc=self.nx.value()*self.ny.value()*self.nz.value();na=nc*len(cfg.BASIS_COORDS)
            f=lambda v:f"{v:.1f}nm" if v<100 else f"{v/1000:.3f}um"
            self.sl.setText(f"-> {f(sz[0])} x {f(sz[1])} x {f(sz[2])}  {nc:,} cells  ~{na:,} atoms")
        except: self.sl.setText("")
    def _bd(self):
        n=self.combo.currentText()
        if n not in MATERIAL_PRESETS: return
        cfg=BCDIConfig(n);cfg.SUPERCELL_MULT=(self.nx.value(),self.ny.value(),self.nz.value());cfg.PARTICLE_SHAPE=self.shp.currentText()
        cfg.STRAIN_TYPE=self.strc.currentText();cfg.STRAIN_MAGNITUDE=self.strm.value()
        if self.dck.isChecked():
            bv=self.db_.value() if self.db_.value()>0 else None
            cfg.DISLOCATION=DislocationConfig(dtype=self.dt.currentText(),pos_frac=(self.dpx.value(),self.dpy.value()),line_dir=self.dd.currentText(),b_angstrom=bv,nu=self.dnu.value())
        if self.lck.isChecked():
            lbv=self.lb.value() if self.lb.value()>0 else None
            cfg.DISLOCATION_LOOP=DislocationLoopConfig(center_frac=(self.lcx.value(),self.lcy.value(),self.lcz.value()),radius_angstrom=self.lr.value(),b_angstrom=lbv,normal=self.ln.currentText(),nu=self.lnu2.value())
        self.config=cfg;self.bb.setEnabled(False);self.nb.setEnabled(False);self.pg.setValue(0);self.lg.clear()
        self._w=BW(cfg);self._w.progress.connect(self.pg.setValue);self._w.log.connect(self.lg.appendPlainText);self._w.done.connect(self._ok);self._w.failed.connect(self._fl);self._w.start()
    def _ok(self,b):
        self.builder=b;self.bb.setEnabled(True);self.nb.setEnabled(True);self.lg.appendPlainText("Rendering...")
        if _HAS_WEB:
            try:
                n=len(b.supercell_positions_ang);asz=3 if n<10000 else(2 if n<40000 else 1)
                html=lattice_html_with_controls(b,50000,asz,show_bonds=self.bc.isChecked(),show_strain=self.stc.isChecked(),show_dislocation=self.dvc.isChecked())
                if html: f=tempfile.NamedTemporaryFile('w',suffix='.html',delete=False,encoding='utf-8');f.write(html);f.close();self.viz.load(QUrl.fromLocalFile(f.name))
            except Exception as e: self.lg.appendPlainText(f"Viz: {e}")
        self.built.emit(self.config,self.builder)
    def _fl(self,m): self.bb.setEnabled(True);self.lg.appendPlainText(f"FAIL: {m}")

# ======================= TAB 2 =======================
class T2(QWidget):
    sim_req=pyqtSignal(dict)
    def __init__(self): super().__init__();self.config=None;self.builder=None;self._rdf=None;self._rt=QTimer();self._rt.setSingleShot(True);self._rt.setInterval(600);self._rt.timeout.connect(self._cr);self._ui()
    def sc(self,c,b): self.config=c;self.builder=b;self._uc();self._cr()
    def _ui(self):
        root=QVBoxLayout(self);root.setContentsMargins(10,10,10,10);root.setSpacing(6)
        h=QHBoxLayout();t=QLabel("Beam & Detector");t.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700");h.addWidget(t,1);self.nb=_nb();h.addWidget(self.nb);root.addLayout(h)
        self.cl=QLabel("Build lattice first.");self.cl.setStyleSheet("color:#f0883e;font-weight:600");root.addWidget(self.cl)
        mh=QHBoxLayout();mh.setSpacing(8);lc=QVBoxLayout();lc.setSpacing(4)
        # Beam
        bb=QGroupBox("X-ray beam");bf=QFormLayout(bb);bf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);bf.setVerticalSpacing(3)
        self.keV=_dbl(.5,100,10,2," keV")
        self.keV.setToolTip(
            "X-ray photon energy in kilo-electron-volts.\n"
            "Common BCDI beamline energies: 7–12 keV.\n"
            "Higher energy → shorter wavelength → tighter 2θ angles."
        )
        self.keV.valueChanged.connect(self._ul);self.keV.valueChanged.connect(lambda:self._rt.start())
        self.ll_=QLabel();self.ll_.setStyleSheet("color:#8b949e;font-size:9pt");self._ul()
        self.pol=_dbl(0,1,1,2)
        self.pol.setToolTip(
            "Polarization factor (0 to 1).\n"
            "1 = full horizontal polarization (typical synchrotron sigma-pol)\n"
            "0 = no polarization correction."
        )
        self.bsz=_dbl(0.1,1000,1,1," um")
        self.bsz.setToolTip(
            "Incident beam size (FWHM) at the sample in micrometers.\n"
            "Typical focused beam: 0.5–3 μm. Affects illumination function\n"
            "in coherent diffraction simulation."
        )
        bf.addRow("Energy:",self.keV);bf.addRow("",self.ll_);bf.addRow("Pol:",self.pol);bf.addRow("Beam size:",self.bsz);lc.addWidget(bb)
        # Detector with preset selector and NX x NY
        db=QGroupBox("Detector");dv=QVBoxLayout(db);dv.setSpacing(3)
        pr=QHBoxLayout();pr.addWidget(QLabel("Preset:"));self.det_combo=QComboBox()
        self.det_combo.setToolTip(
            "Pre-configured detector. Auto-fills pixel count and pixel size.\n"
            "  • Maxipix (ID01):  516×516, 55 μm pixels\n"
            "  • Eiger 2M:        1062×1028, 75 μm pixels\n"
            "  • Custom:          set values manually"
        )
        for k in DETECTOR_PRESETS: self.det_combo.addItem(k)
        self.det_combo.currentTextChanged.connect(self._det_preset);pr.addWidget(self.det_combo,1);dv.addLayout(pr)
        df=QFormLayout();df.setLabelAlignment(Qt.AlignmentFlag.AlignRight);df.setVerticalSpacing(3)
        self.det_d=_dbl(.05,10,.5,3," m")
        self.det_d.setToolTip(
            "Sample-to-detector distance in meters.\n"
            "Determines q-space resolution: closer = smaller q range, farther =\n"
            "finer q sampling. Typical BCDI: 0.5–2 m."
        )
        # NX x NY row
        pxr=QHBoxLayout();self.det_nx=QSpinBox();self.det_nx.setRange(32,4096);self.det_nx.setValue(128)
        self.det_nx.setToolTip("Number of detector pixels in the horizontal (X) direction.")
        pxr.addWidget(self.det_nx);pxr.addWidget(QLabel("x"));self.det_ny=QSpinBox();self.det_ny.setRange(32,4096);self.det_ny.setValue(128)
        self.det_ny.setToolTip("Number of detector pixels in the vertical (Y) direction.")
        pxr.addWidget(self.det_ny)
        self.det_sz=_dbl(1,500,55,1," um")
        self.det_sz.setToolTip(
            "Physical pixel pitch in micrometers.\n"
            "Maxipix: 55 μm.  Eiger 2M: 75 μm.\n"
            "Smaller pixels → finer angular resolution per pixel."
        )
        df.addRow("Distance:",self.det_d);df.addRow("Pixels:",pxr);df.addRow("Pixel size:",self.det_sz)
        # Rocking: "Steps" label (= number of rocking curve angular positions)
        self.rock=_dbl(.0001,1,.00315,5," deg");self.rock.setDecimals(5)
        self.rock.setToolTip(
            "Angular step size between rocking curve points, in degrees.\n"
            "Sets the q-resolution along the rocking direction.\n"
            "Typical: 0.001–0.01°. Smaller = finer rocking sampling but more frames."
        )
        self.steps=QSpinBox();self.steps.setRange(8,2048);self.steps.setValue(128)
        self.steps.setToolTip(
            "Number of rocking curve points (i.e., number of detector frames\n"
            "in the scan). Total rocking range = step × steps.\n"
            "Typical: 50–200 for full Bragg peak coverage."
        )
        df.addRow("Rock step:",self.rock);df.addRow("Steps:",self.steps)
        # Oversampling with tooltip
        self.os=_dbl(1,20,5,1);self.os.setToolTip(
            "Oversampling ratio: simulation grid = OS × Nyquist.\n"
            "Higher = finer q-resolution but slower.\n"
            "Typical BCDI: 3–8.  Reconstruction is only possible if OS > 2."
        )
        os_row=QHBoxLayout();os_row.addWidget(self.os);os_lbl=QLabel("(oversampling)");os_lbl.setStyleSheet("color:#8b949e;font-size:8pt");os_row.addWidget(os_lbl)
        df.addRow("OS ratio:",os_row)
        dv.addLayout(df);lc.addWidget(db)
        # Coherence
        cb=QGroupBox("Coherence");cf=QFormLayout(cb);cf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);cf.setVerticalSpacing(3)
        self.sh=_dbl(0,10000,300,1," um")
        self.sh.setToolTip(
            "Horizontal source size (FWHM) at the source point, in micrometers.\n"
            "Affects horizontal transverse coherence at the sample.\n"
            "Typical synchrotron: 100–500 μm. Set to 0 to disable coherence model."
        )
        self.sv=_dbl(0,10000,10,1," um")
        self.sv.setToolTip(
            "Vertical source size (FWHM) at the source point, in micrometers.\n"
            "Affects vertical transverse coherence at the sample.\n"
            "Typical synchrotron: 5–50 μm (smaller than H — synchrotrons are\n"
            "vertically coherent by design)."
        )
        self.sd=_dbl(.1,500,50,1," m")
        self.sd.setToolTip(
            "Source-to-sample distance in meters.\n"
            "Longer distance → better transverse coherence (larger coherence\n"
            "length at the sample). Typical: 30–80 m."
        )
        self.bw=_dbl(1e-6,.1,1e-4,6);self.bw.setDecimals(6)
        self.bw.setToolTip(
            "Relative energy bandwidth ΔE/E of the monochromator.\n"
            "Sets the longitudinal coherence length.\n"
            "Si(111) monochromator: ~1.3×10⁻⁴.  Multilayer: 10⁻² to 10⁻³."
        )
        cf.addRow("Src H:",self.sh);cf.addRow("Src V:",self.sv);cf.addRow("Src D:",self.sd);cf.addRow("dE/E:",self.bw);lc.addWidget(cb);lc.addStretch();mh.addLayout(lc)
        # Center: reflection chart
        cc=QVBoxLayout();cc.setSpacing(4);cc.addWidget(QLabel("Reflection intensities"))
        with matplotlib.rc_context(MPL_DARK): self.rf=Figure(figsize=(4,5),tight_layout=True);self.rc_=FigureCanvas(self.rf)
        self.rc_.setMinimumWidth(250);cc.addWidget(self.rc_,1);mh.addLayout(cc,1)
        # Right: reflection table + angle input
        rc=QVBoxLayout();rc.setSpacing(4);rc.addWidget(QLabel("Reflection selection"))
        self.rt=QTableWidget(0,7);self.rt.setHorizontalHeaderLabels(["hkl","d(A)","2th","th_B","|F|","I_norm","BCDI"])
        self.rt.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.rt.setAlternatingRowColors(True);self.rt.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rt.setMinimumWidth(320);rc.addWidget(self.rt,1)
        hr=QHBoxLayout();hr.addWidget(QLabel("hkl:"));self.he=QLineEdit();self.he.setPlaceholderText("e.g. 1 1 1");hr.addWidget(self.he,1);rc.addLayout(hr)
        # Custom angle input
        ar=QHBoxLayout();ar.addWidget(QLabel("Custom 2th:"));self.custom_angle=_dbl(0,180,0,3," deg");self.custom_angle.setSpecialValueText("auto");ar.addWidget(self.custom_angle,1);rc.addLayout(ar)
        mh.addLayout(rc,1);root.addLayout(mh,1)
        self.sb_=QPushButton("Simulate");self.sb_.setStyleSheet("background:#1f6feb;min-height:34px;font-size:11pt");self.sb_.setEnabled(False);self.sb_.clicked.connect(self._rn);root.addWidget(self.sb_)
        self.pg=QProgressBar();root.addWidget(self.pg);self.lg=QPlainTextEdit();self.lg.setReadOnly(True);self.lg.setMaximumHeight(55);root.addWidget(self.lg)
    def _det_preset(self,name):
        if name in DETECTOR_PRESETS and name!='Custom':
            d=DETECTOR_PRESETS[name];self.det_nx.setValue(d['nx']);self.det_ny.setValue(d['ny']);self.det_sz.setValue(d['pixel_um'])
    def _ul(self): lam=BCDIConfig.HC_KEV_ANG/self.keV.value() if self.keV.value()>0 else 0;self.ll_.setText(f"lam = {lam:.4f} A")
    def _uc(self):
        if self.config is None: self.cl.setText("Build lattice first.");self.cl.setStyleSheet("color:#f0883e;font-weight:600");self.sb_.setEnabled(False)
        else:
            sz=self.config.particle_size_nm;n=len(self.builder.supercell_positions_ang) if self.builder else 0
            self.cl.setText(f"{self.config.MATERIAL_NAME} ({self.config.SPACE_GROUP}) {sz[0]:.1f}x{sz[1]:.1f}x{sz[2]:.1f}nm {n:,}atoms");self.cl.setStyleSheet("color:#3fb950;font-weight:600");self.sb_.setEnabled(True)
    def _cr(self):
        if self.config is None or self.builder is None: return
        try:
            c2=BCDIConfig(self.config.MATERIAL_NAME);c2.SUPERCELL_MULT=self.config.SUPERCELL_MULT;c2.BEAM_ENERGY_KEV=self.keV.value();c2.POLARIZATION_FACTOR=self.pol.value()
            rc=ReflectionCalculator(c2,self.builder);df=rc.calculate();self._rdf=df;self.rt.setRowCount(len(df))
            for i,(_,r) in enumerate(df.iterrows()):
                for j,txt in enumerate([r.get('hkl_display',r['hkl_str']),f"{r['d_Ang']:.4f}",f"{r['2theta']:.2f}",f"{r['theta_B']:.2f}",f"{r['|F|']:.1f}",f"{r['I_norm']:.3f}",""]):
                    it=QTableWidgetItem(txt);it.setFlags(it.flags()&~Qt.ItemFlag.ItemIsEditable)
                    if r['BCDI_flag']: it.setForeground(QColor("#3fb950"))
                    self.rt.setItem(i,j,it)
                if r['BCDI_flag']:
                    star=QTableWidgetItem("★");star.setFlags(star.flags()&~Qt.ItemFlag.ItemIsEditable);star.setForeground(QColor("#3fb950"));star.setFont(QFont("Segoe UI",14));star.setTextAlignment(Qt.AlignmentFlag.AlignCenter);self.rt.setItem(i,6,star)
            bcdi=df[df['BCDI_flag']]
            if len(bcdi): self.rt.selectRow(bcdi.index[0])
            elif len(df): self.rt.selectRow(0)
            self._dc(df)
        except Exception as e: self.lg.appendPlainText(f"Refl: {e}")
    def _dc(self,df):
        with matplotlib.rc_context(MPL_DARK):
            self.rf.clear();ax=self.rf.add_subplot(111);ax.set_facecolor('#000005')
            if len(df)==0: self.rc_.draw();return
            colors=['#3fb950' if b else '#4f98a3' for b in df['BCDI_flag']]
            ax.bar(range(len(df)),df['I_norm'].values,color=colors,width=0.7)
            ax.set_xticks(range(len(df)));ax.set_xticklabels([f"{l}\n{t:.1f}" for l,t in zip(df['hkl_display'],df['2theta'])],fontsize=6)
            ax.set_ylabel('I_norm',fontsize=8);ax.set_title(f'{self.keV.value():.1f} keV',fontsize=9);ax.tick_params(labelsize=6)
        self.rc_.draw()
    def _gh(self):
        txt=self.he.text().strip()
        if txt:
            try: p=[int(s) for s in txt.replace(',',' ').split()];assert len(p)==3;return tuple(p)
            except: QMessageBox.warning(self,"Bad","Enter 3 ints");return'BAD'
        sel=self.rt.selectedItems()
        if sel and self._rdf is not None:
            row=sel[0].row()
            if row<len(self._rdf): return self._rdf.iloc[row]['hkl']
        return None
    def _rn(self):
        if self.config is None: return
        hkl=self._gh()
        if hkl=='BAD': return
        c=self.config;c.BEAM_ENERGY_KEV=self.keV.value();c.POLARIZATION_FACTOR=self.pol.value();c.BEAM_SIZE_UM=self.bsz.value()
        c.SAMPLE_DETECTOR_DISTANCE_M=self.det_d.value();c.DETECTOR_NX=self.det_nx.value();c.DETECTOR_NY=self.det_ny.value()
        c.DETECTOR_PIXEL_UM=self.det_sz.value();c.ROCKING_STEP_DEG=self.rock.value();c.ROCKING_STEPS=self.steps.value()
        c.SOURCE_SIZE_H_UM=self.sh.value();c.SOURCE_SIZE_V_UM=self.sv.value();c.SOURCE_DISTANCE_M=self.sd.value();c.MONOCHR_BANDWIDTH=self.bw.value();c.TARGET_OVERSAMPLING=self.os.value()
        self.lg.clear()
        # Custom angle: compute offset from nearest Bragg peak
        custom_2th=self.custom_angle.value()
        if custom_2th>0.01:
            # Find nearest reflection's 2theta
            if self._rdf is not None and len(self._rdf)>0:
                nearest=self._rdf.iloc[(self._rdf['2theta']-custom_2th).abs().argsort().iloc[0]]
                c.ANGULAR_OFFSET_DEG=custom_2th-nearest['2theta']
                hkl=nearest['hkl']  # use nearest reflection as base
                self.lg.appendPlainText(f"Custom 2th={custom_2th:.3f}, nearest={nearest['hkl_display']} at {nearest['2theta']:.3f}, offset={c.ANGULAR_OFFSET_DEG:.4f} deg")
            else:
                c.ANGULAR_OFFSET_DEG=0.0
        else:
            c.ANGULAR_OFFSET_DEG=0.0
        self.sb_.setEnabled(False);self.nb.setEnabled(False);self.pg.setValue(0);self.sim_req.emit({'config':c,'builder':self.builder,'hkl':hkl})

# ======================= TAB 3 =======================
class T3(QWidget):
    def __init__(self): super().__init__();self._s=self._r=self._c=None;self._ui()
    def _ui(self):
        root=QVBoxLayout(self);root.setContentsMargins(12,12,12,12);root.setSpacing(6)
        h=QHBoxLayout();h.addWidget(QLabel("Diffraction Results"));h.itemAt(0).widget().setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700")
        h.addSpacing(15);h.addWidget(QLabel("Effects:"));h.itemAt(2).widget().setStyleSheet("color:#8b949e;font-size:9pt")
        self.np_=QCheckBox("Poisson");self.nr=QCheckBox("Readout");self.na=QCheckBox("Air");self.nd=QCheckBox("Dead")
        for c in(self.np_,self.nr,self.na,self.nd): h.addWidget(c)
        self.rb=QPushButton("Re-render");self.rb.setEnabled(False);self.rb.setStyleSheet("background:#4f98a3;padding:4px 10px;min-height:20px;font-size:9pt");self.rb.clicked.connect(self._rr);h.addWidget(self.rb)
        h.addStretch();root.addLayout(h)
        self.il=QLabel("Run simulation.");self.il.setStyleSheet("color:#8b949e;font-size:9pt");root.addWidget(self.il)
        sp=QSplitter(Qt.Orientation.Horizontal);lp=QWidget();ll=QVBoxLayout(lp);ll.setContentsMargins(0,0,0,0);ll.addWidget(QLabel("3D Bragg Peak (measure: check box in controls)"))
        if _HAS_WEB: self.bv=QWebEngineView();self.bv.setHtml(_ph("3D after sim."))
        else: self.bv=QLabel("No WebEngine")
        self.bv.setMinimumWidth(520);ll.addWidget(self.bv,1);sp.addWidget(lp)
        rp=QWidget();rl=QVBoxLayout(rp);rl.setContentsMargins(0,0,0,0);rl.setSpacing(4);rl.addWidget(QLabel("2D Slices"))
        with matplotlib.rc_context(MPL_DARK): self.f1=Figure(figsize=(5,3.4),dpi=150,tight_layout=True);self.f2=Figure(figsize=(5,3.4),dpi=150,tight_layout=True)
        self.c1=FigureCanvas(self.f1);self.c2=FigureCanvas(self.f2)
        for c in(self.c1,self.c2):
            c.setMinimumHeight(230);fr=QFrame();fr.setFrameShape(QFrame.Shape.StyledPanel);fl=QVBoxLayout(fr);fl.setContentsMargins(0,0,0,0)
            tb=NavigationToolbar(c,fr);tb.setStyleSheet("background:#161b22");fl.addWidget(tb);fl.addWidget(c,1);rl.addWidget(fr,1)
        sp.addWidget(rp);sp.setSizes([600,500]);root.addWidget(sp,1)
        # Export button at bottom
        eb=QHBoxLayout();eb.addStretch()
        self.exp_btn=QPushButton("Export measurement (.npz / .h5)");self.exp_btn.setEnabled(False)
        self.exp_btn.setStyleSheet("background:#1f6feb;padding:8px 20px;min-height:28px");self.exp_btn.clicked.connect(self._export);eb.addWidget(self.exp_btn)
        eb.addStretch();root.addLayout(eb)
    def _gn(self):
        o={}
        if self.np_.isChecked(): o['poisson']=True
        if self.nr.isChecked(): o['readout_noise']=5.0
        if self.na.isChecked(): o['air_scatter']=100.0
        if self.nd.isChecked(): o['dead_pixels_frac']=0.002
        return o or None
    def _rr(self):
        if self._s: self._di(self._s,self._r,self._c,self._gn())
    def _export(self):
        if self._s is None: return
        path,_=QFileDialog.getSaveFileName(self,"Export measurement","bcdi_measurement","NumPy (*.npz);;HDF5 (*.h5)")
        if not path: return
        try:
            export_measurement(self._s,self._r,self._c,path,self._gn());QMessageBox.information(self,"Exported",f"Saved to {path}")
        except Exception as e: QMessageBox.critical(self,"Export Error",str(e))
    def display(self,sim,refl,cfg): self._s=sim;self._r=refl;self._c=cfg;self.rb.setEnabled(True);self.exp_btn.setEnabled(True);self._di(sim,refl,cfg,self._gn())
    def _di(self,sim,refl,cfg,no):
        self.il.setText(f"{cfg.MATERIAL_NAME} {refl['hkl_str']} lam={cfg.wavelength_angstrom:.3f}A 2th={np.degrees(refl['two_theta_rad']):.3f} beam={cfg.BEAM_SIZE_UM:.1f}um det={cfg.DETECTOR_NX}x{cfg.DETECTOR_NY}")
        if _HAS_WEB:
            try:
                html=bragg_html_with_controls(sim,2,1e-4,no)
                if html: f=tempfile.NamedTemporaryFile('w',suffix='.html',delete=False,encoding='utf-8');f.write(html);f.close();self.bv.load(QUrl.fromLocalFile(f.name))
            except Exception as e: print(f"3D: {e}")
        vol=sim.diff_volume.copy()
        if no: vol=add_experimental_noise(vol,**no)
        qx=sim.q_grids['qx'];qy=sim.q_grids['qy'];qz=sim.q_grids['qz'];N=vol.shape[0];cn=N//2;lmx=np.log10(vol.max()+1.);lmn=max(0,lmx-6)
        # Up-sample slice data with cubic spline interpolation BEFORE imshow.
        # The simulation cube has unequal effective sampling per axis (the
        # rocking direction may have fewer real samples than the detector
        # axes), so a raw imshow shows blocky stripes. Zooming the 2D slice
        # 4x with cubic interpolation gives a visually clean rendering
        # without changing the underlying physics.
        from scipy.ndimage import zoom as _zoom
        zoom_factor = 4
        for fig,title,data,qh,qv,xl,yl in [(self.f1,"qx-qy",vol[:,:,cn],qx,qy,"qx","qy"),(self.f2,"qx-qz",vol[:,cn,:],qx,qz,"qx","qz")]:
            with matplotlib.rc_context(MPL_DARK):
                fig.clear();ax=fig.add_subplot(111);ax.set_facecolor('#000005')
                # Cubic interpolation in linear space first, then take log
                data_smooth = _zoom(data, zoom_factor, order=3, mode='nearest')
                # Clip to non-negative (cubic can over/undershoot)
                data_smooth = np.maximum(data_smooth, 0)
                ld=np.log10(data_smooth.T+1.)
                im=ax.imshow(ld,origin='lower',cmap='jet',aspect='auto',extent=[qh[0],qh[-1],qv[0],qv[-1]],vmin=lmn,vmax=lmx,interpolation='bilinear')
                tA=cfg.particle_size_angstrom;dq=2*np.pi/np.mean(tA[:2]);qz_=8*dq;ax.set_xlim(-qz_,qz_);ax.set_ylim(-qz_,qz_)
                div=make_axes_locatable(ax);cax=div.append_axes('right',size='4%',pad=.05);fig.colorbar(im,cax=cax)
                ax.set_title(title,fontsize=9);ax.set_xlabel(xl,fontsize=8);ax.set_ylabel(yl,fontsize=8);ax.tick_params(labelsize=7)
        self.c1.draw();self.c2.draw()

# ======================= REPORTS & SUGGESTIONS =======================
class ReportsDialog(QWidget):
    """Small dialog: user enters email and message, sent to saidisoufiane@hotmail.com"""
    RECIPIENT = "saidisoufiane@hotmail.com"
    def __init__(self,parent=None):
        super().__init__(parent,Qt.WindowType.Dialog)
        self.setWindowTitle("Reports & Suggestions");self.setFixedSize(480,360);self.setStyleSheet(QSS)
        v=QVBoxLayout(self);v.setContentsMargins(16,16,16,16);v.setSpacing(10)
        title=QLabel("Reports & Suggestions");title.setStyleSheet("color:#4f98a3;font-size:14pt;font-weight:700");v.addWidget(title)
        info=QLabel("<span style='color:#8b949e;font-size:9pt'>"
                    "Both fields are required. "
                    "Your default mail client will open with the message ready to send."
                    "</span>")
        info.setWordWrap(True);v.addWidget(info)
        v.addWidget(QLabel("Your email:"));self.email_le=QLineEdit();self.email_le.setPlaceholderText("you@example.com");v.addWidget(self.email_le)
        v.addWidget(QLabel("Message:"));self.msg_te=QPlainTextEdit();self.msg_te.setPlaceholderText("Describe the issue or suggestion...");self.msg_te.setMinimumHeight(140);v.addWidget(self.msg_te,1)
        self.status=QLabel("");self.status.setWordWrap(True);self.status.setStyleSheet("font-size:9pt");v.addWidget(self.status)
        br=QHBoxLayout();br.addStretch()
        cancel=QPushButton("Cancel");cancel.setStyleSheet("background:#30363d;padding:7px 14px");cancel.clicked.connect(self.close);br.addWidget(cancel)
        send=QPushButton("Send");send.setStyleSheet("background:#238636;padding:7px 18px;font-weight:600");send.clicked.connect(self._send);br.addWidget(send)
        v.addLayout(br)
    def _validate_email(self,e):
        import re
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$",e))
    def _send(self):
        email=self.email_le.text().strip();msg=self.msg_te.toPlainText().strip()
        if not email:
            self.status.setText("<span style='color:#da3633'>Email is required.</span>");return
        if not self._validate_email(email):
            self.status.setText("<span style='color:#da3633'>Please enter a valid email address.</span>");return
        if not msg:
            self.status.setText("<span style='color:#da3633'>Message is required.</span>");return
        # Open the user's default mail client via mailto:
        import urllib.parse,webbrowser
        subject=urllib.parse.quote(f"BCDI Software - Report from {email}")
        body=urllib.parse.quote(f"From: {email}\n\n{msg}")
        url=f"mailto:{self.RECIPIENT}?subject={subject}&body={body}"
        try:
            webbrowser.open(url,new=1)
            self.status.setText(
                f"<span style='color:#3fb950'>\u2713 Opening your mail client. "
                f"Click 'Send' there to deliver the message to {self.RECIPIENT}.</span>"
            )
            QTimer.singleShot(2500,self.close)
        except Exception as e:
            # Fallback: copy mailto URL to clipboard
            try:
                cb=QApplication.clipboard();cb.setText(f"To: {self.RECIPIENT}\nFrom: {email}\n\n{msg}")
                self.status.setText(
                    f"<span style='color:#f0883e'>Could not open mail client automatically. "
                    f"The message has been copied to clipboard \u2014 please paste it into "
                    f"an email to {self.RECIPIENT}.</span>"
                )
            except Exception as e2:
                self.status.setText(f"<span style='color:#da3633'>Failed: {e2}</span>")


def _make_reports_button(parent):
    """Create a small 'Reports & Suggestions' button anchored bottom-right.
    Note: Qt treats `&` in button labels as a keyboard accelerator marker
    (the next character becomes Alt+letter). To display a literal `&`,
    we use `&&`."""
    btn=QPushButton("Reports && Suggestions",parent)
    # 60% smaller than the previous version: smaller padding, smaller font, smaller height
    btn.setStyleSheet(
        "background:#1f6feb;color:#fff;padding:1px 6px;border-radius:3px;"
        "font-size:7pt;font-weight:600;border:none"
    )
    btn.setToolTip("Send feedback or report a bug")
    btn.setFixedHeight(16)
    btn.adjustSize()
    btn._dialog=None
    def _open():
        if btn._dialog is None or not btn._dialog.isVisible():
            btn._dialog=ReportsDialog(parent)
        btn._dialog.show();btn._dialog.raise_();btn._dialog.activateWindow()
    btn.clicked.connect(_open)
    return btn


# ======================= SIMULATION WINDOW =======================
class MW_Sim(QMainWindow):
    """Simulation-only window: Material, Beam, Results tabs (no numbering)."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle("BCDI Simulation");self.resize(1500,960);self.setMinimumSize(1200,800)
        central=QWidget();vl=QVBoxLayout(central);vl.setContentsMargins(0,0,0,0);vl.setSpacing(0)
        self.tabs=QTabWidget();self.tabs.setDocumentMode(True)
        self.t1=T1();self.t2=T2();self.t3=T3()
        self.tabs.addTab(self.t1,"Material");self.tabs.addTab(self.t2,"Beam");self.tabs.addTab(self.t3,"Results")
        self.tabs.setTabEnabled(1,False);self.tabs.setTabEnabled(2,False)
        vl.addWidget(self.tabs,1)
        self.setCentralWidget(central)
        sb=QStatusBar();self.setStatusBar(sb);sb.showMessage("Build a lattice to begin.")
        self.t1.built.connect(self._ob);self.t1.nb.clicked.connect(lambda:self.tabs.setCurrentIndex(1))
        self.t2.sim_req.connect(self._os);self.t2.nb.clicked.connect(lambda:self.tabs.setCurrentIndex(2))
        self._sw=None
        # Reports button bottom-right
        self.report_btn=_make_reports_button(self)
        self._reposition_reports()
    def resizeEvent(self,e):
        super().resizeEvent(e);self._reposition_reports()
    def _reposition_reports(self):
        if hasattr(self,'report_btn'):
            margin=10;sb_h=self.statusBar().height() if self.statusBar() else 0
            self.report_btn.move(self.width()-self.report_btn.width()-margin,
                                  self.height()-self.report_btn.height()-sb_h-margin)
            self.report_btn.raise_()
    def _ob(self,c,b): self.t2.sc(c,b);self.tabs.setTabEnabled(1,True)
    def _os(self,p):
        self._sw=SW(p['config'],p['builder'],p['hkl']);self._sw.progress.connect(self.t2.pg.setValue);self._sw.log.connect(self.t2.lg.appendPlainText)
        self._sw.done.connect(self._sd);self._sw.failed.connect(self._sf);self._sw.start()
    def _sd(self,s,r,d): self.t2.sb_.setEnabled(True);self.t2.nb.setEnabled(True);self.t3.display(s,r,s.config);self.tabs.setTabEnabled(2,True)
    def _sf(self,m): self.t2.sb_.setEnabled(True);self.t2.lg.appendPlainText(f"FAIL: {m}")


# ======================= DATA ANALYSIS WINDOW =======================
class MW_Analysis(QMainWindow):
    """Data analysis window: Generate Data, AutoPhaseNN, CDI NN, Reconstruction, 3D Viewer."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle("BCDI Data Analysis");self.resize(1500,960);self.setMinimumSize(1200,800)
        if not _HAS_NN:
            QMessageBox.critical(self,"Missing module",
                "The data analysis module (nn_gui_tabs) is not available. "
                "Make sure all NN-related files are present.")
            QTimer.singleShot(0,self.close);return
        central=QWidget();vl=QVBoxLayout(central);vl.setContentsMargins(0,0,0,0);vl.setSpacing(0)
        self.tabs=QTabWidget();self.tabs.setDocumentMode(True)
        self.tg=T_Gen();self.t4=T4();self.t4s=T4_Sup();self.t5=T5();self.t6=T6()
        self.tabs.addTab(self.tg,"Generate Data")
        self.tabs.addTab(self.t4,"AutoPhase_NN")
        self.tabs.addTab(self.t4s,"CDI_NN")
        self.tabs.addTab(self.t5,"Reconstruction")
        self.tabs.addTab(self.t6,"3D Viewer")
        self.t5.recon_done.connect(self.t6.set_result)
        vl.addWidget(self.tabs,1)
        self.setCentralWidget(central)
        sb=QStatusBar();self.setStatusBar(sb);sb.showMessage("Generate training data, train models, then run reconstruction.")
        # Reports button bottom-right
        self.report_btn=_make_reports_button(self)
        self._reposition_reports()
    def resizeEvent(self,e):
        super().resizeEvent(e);self._reposition_reports()
    def _reposition_reports(self):
        if hasattr(self,'report_btn'):
            margin=10;sb_h=self.statusBar().height() if self.statusBar() else 0
            self.report_btn.move(self.width()-self.report_btn.width()-margin,
                                  self.height()-self.report_btn.height()-sb_h-margin)
            self.report_btn.raise_()


# ======================= LAUNCHER (small first window) =======================
class Launcher(QMainWindow):
    """Launcher window with the CDI-ST logo + two mode buttons + Exit."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CDI-ST")
        # Sized so the 678x340 logo fits at a reasonable scale
        self.setFixedSize(560, 540)
        self.sim_window = None
        self.analysis_window = None
        central = QWidget()
        central.setStyleSheet("background:#0d1117")
        v = QVBoxLayout(central)
        v.setContentsMargins(20, 16, 20, 14)
        v.setSpacing(12)
        # Logo image at the top (matches splash screen)
        logo = _logo_path()
        if logo is not None:
            pix = QPixmap(logo)
            if not pix.isNull():
                # Scale so it fits in the launcher width comfortably
                pix = pix.scaledToWidth(520, Qt.TransformationMode.SmoothTransformation)
                logo_label = QLabel()
                logo_label.setPixmap(pix)
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                logo_label.setStyleSheet("background:transparent")
                v.addWidget(logo_label)
            else:
                self._add_text_header(v)
        else:
            self._add_text_header(v)
        v.addSpacing(6)
        choose = QLabel("Choose a module:")
        choose.setStyleSheet("color:#e6edf3;font-size:10pt;background:transparent")
        v.addWidget(choose)
        # Mode buttons (no tooltips — labels are self-explanatory)
        sim_btn = QPushButton("BCDI Simulation")
        sim_btn.setStyleSheet(
            "background:#238636;color:#fff;padding:14px;border-radius:6px;"
            "font-size:11pt;font-weight:600;text-align:center"
        )
        sim_btn.setMinimumHeight(48)
        sim_btn.clicked.connect(self._open_sim)
        v.addWidget(sim_btn)
        ana_btn = QPushButton("BCDI Data Analysis")
        ana_btn.setStyleSheet(
            "background:#1f6feb;color:#fff;padding:14px;border-radius:6px;"
            "font-size:11pt;font-weight:600;text-align:center"
        )
        ana_btn.setMinimumHeight(48)
        ana_btn.clicked.connect(self._open_analysis)
        v.addWidget(ana_btn)
        v.addSpacing(4)
        # Exit button (smaller, red)
        exit_btn = QPushButton("Exit")
        exit_btn.setStyleSheet(
            "background:#da3633;color:#fff;padding:8px;border-radius:5px;"
            "font-size:10pt;font-weight:600"
        )
        exit_btn.clicked.connect(self._exit_app)
        v.addWidget(exit_btn)
        v.addStretch()
        self.setCentralWidget(central)
        # Software credit bottom-right (small grey)
        self.credit = QLabel("Software developed by Soufiane SAIDI", self)
        self.credit.setStyleSheet(
            "color:#6e7681;font-size:8pt;background:transparent"
        )
        self.credit.adjustSize()
        self._reposition_credit()
    def _add_text_header(self, layout):
        """Fallback header if the logo image is missing — text only."""
        title = QLabel("CDI-ST")
        title.setStyleSheet(
            "color:#e6edf3;font-size:32pt;font-weight:700;"
            "background:transparent;letter-spacing:2px"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        sub = QLabel("Coherent Diffraction Imaging Simulation Tools")
        sub.setStyleSheet("color:#8b949e;font-size:10pt;background:transparent")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reposition_credit()
    def _reposition_credit(self):
        if hasattr(self, 'credit'):
            margin = 8
            self.credit.move(
                self.width() - self.credit.width() - margin,
                self.height() - self.credit.height() - margin
            )
            self.credit.raise_()
    def _open_sim(self):
        if self.sim_window is None or not self.sim_window.isVisible():
            self.sim_window = MW_Sim(); self.sim_window.show()
        else:
            self.sim_window.raise_(); self.sim_window.activateWindow()
    def _open_analysis(self):
        if self.analysis_window is None or not self.analysis_window.isVisible():
            self.analysis_window = MW_Analysis(); self.analysis_window.show()
        else:
            self.analysis_window.raise_(); self.analysis_window.activateWindow()
    def _exit_app(self):
        if self.sim_window is not None: self.sim_window.close()
        if self.analysis_window is not None: self.analysis_window.close()
        QApplication.quit()


# ======================= SPLASH SCREEN =======================
def _logo_path():
    """Return the path to the CDI-ST logo, looking in several plausible places."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "CDI_ST_logo.png"),
        os.path.join(here, "CDI_NN_logo.png"),
        os.path.join(here, "logo.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


class CDIStSplash(QSplashScreen):
    """
    Splash screen showing the CDI-ST logo with a loading bar underneath.
    Matches the dark theme of the rest of the application.
    """
    def __init__(self):
        # Build the splash pixmap: logo image + space below for loading bar
        logo = _logo_path()
        if logo is not None:
            base_pix = QPixmap(logo)
            if base_pix.isNull():
                base_pix = self._fallback_pixmap()
        else:
            base_pix = self._fallback_pixmap()
        # Scale to a reasonable splash size if needed
        if base_pix.width() > 700:
            base_pix = base_pix.scaledToWidth(700, Qt.TransformationMode.SmoothTransformation)
        # Create a final pixmap with extra space for the loading bar + status text
        BAR_H = 64  # extra height below the logo for bar + text
        full = QPixmap(base_pix.width(), base_pix.height() + BAR_H)
        full.fill(QColor("#0d1117"))   # dark background matching app theme
        p = QPainter(full)
        p.drawPixmap(0, 0, base_pix)
        p.end()
        super().__init__(full, Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(full.size())
        self._logo_h = base_pix.height()
        self._bar_h = BAR_H
        self._progress = 0
        self._status = "Loading..."
        # Loading bar overlay (drawn in drawContents)
    def _fallback_pixmap(self):
        """If the logo file is missing, build a simple text-based banner."""
        pix = QPixmap(640, 280)
        pix.fill(QColor("#0d1117"))
        p = QPainter(pix)
        # Title
        p.setPen(QColor("#e6edf3"))
        f = QFont(); f.setPointSize(48); f.setBold(True); f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4)
        p.setFont(f)
        p.drawText(QRect(0, 50, 640, 80), Qt.AlignmentFlag.AlignCenter, "CDI-ST")
        # Subtitle
        p.setPen(QColor("#8b949e"))
        f2 = QFont(); f2.setPointSize(13)
        p.setFont(f2)
        p.drawText(QRect(0, 150, 640, 30), Qt.AlignmentFlag.AlignCenter,
                   "Coherent Diffraction Imaging Simulation Tools")
        p.end()
        return pix
    def set_progress(self, value, status=""):
        self._progress = max(0, min(100, int(value)))
        if status:
            self._status = status
        self.repaint()
    def drawContents(self, painter):
        # Draw loading bar + status text in the strip below the logo
        w = self.width()
        bar_y_top = self._logo_h + 18
        bar_x = 40
        bar_w = w - 80
        bar_h = 8
        # Track
        painter.setBrush(QBrush(QColor("#161b22")))
        painter.setPen(QPen(QColor("#30363d"), 1))
        painter.drawRoundedRect(bar_x, bar_y_top, bar_w, bar_h, 3, 3)
        # Filled portion
        if self._progress > 0:
            painter.setBrush(QBrush(QColor("#4f98a3")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_x, bar_y_top, int(bar_w * self._progress / 100.0), bar_h, 3, 3)
        # Status text
        painter.setPen(QColor("#8b949e"))
        f = QFont(); f.setPointSize(9)
        painter.setFont(f)
        painter.drawText(
            QRect(bar_x, bar_y_top + bar_h + 6, bar_w, 18),
            Qt.AlignmentFlag.AlignCenter,
            self._status
        )


# ======================= MAIN =======================
def main():
    app = QApplication(sys.argv); app.setStyle("Fusion"); pal = QPalette()
    for r, c in [(QPalette.ColorRole.Window, "#0d1117"),
                 (QPalette.ColorRole.WindowText, "#e6edf3"),
                 (QPalette.ColorRole.Base, "#0d1117"),
                 (QPalette.ColorRole.AlternateBase, "#161b22"),
                 (QPalette.ColorRole.Text, "#e6edf3"),
                 (QPalette.ColorRole.Button, "#21262d"),
                 (QPalette.ColorRole.ButtonText, "#e6edf3"),
                 (QPalette.ColorRole.Highlight, "#1f6feb"),
                 (QPalette.ColorRole.HighlightedText, "#ffffff")]:
        pal.setColor(r, QColor(c))
    app.setPalette(pal); app.setStyleSheet(QSS)
    # Show splash screen first
    splash = CDIStSplash()
    splash.show(); app.processEvents()
    # Simulate a few-stage loading sequence so the user sees the bar fill
    import time
    stages = [
        (10, "Initializing Qt..."),
        (30, "Loading core modules..."),
        (55, "Loading NN modules..."),
        (80, "Preparing UI..."),
        (100, "Ready"),
    ]
    for pct, msg in stages:
        splash.set_progress(pct, msg)
        app.processEvents()
        time.sleep(0.18)
    # Build and show launcher
    win = Launcher()
    win.show()
    splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
