import sys,tempfile,os
import numpy as np
from collections import Counter
from PyQt6.QtCore import Qt,QThread,pyqtSignal,QUrl,QSize,QTimer
from PyQt6.QtGui import QPalette,QColor,QFont
from PyQt6.QtWidgets import (QApplication,QMainWindow,QWidget,QTabWidget,QVBoxLayout,QHBoxLayout,
    QFormLayout,QGridLayout,QGroupBox,QLabel,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,
    QLineEdit,QTableWidget,QTableWidgetItem,QHeaderView,QProgressBar,QMessageBox,QStatusBar,
    QSplitter,QPlainTextEdit,QFrame,QCheckBox,QScrollArea,QFileDialog)
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView; _HAS_WEB=True
except: _HAS_WEB=False
import matplotlib; matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
if __package__:                      # installed as cdi_st
    from .bcdi_core import (MATERIAL_PRESETS,DETECTOR_PRESETS,BCDIConfig,CrystalBuilder,ReflectionCalculator,
    BCDISimulator,DislocationConfig,DislocationLoopConfig,lattice_html_with_controls,bragg_html_with_controls,
    add_experimental_noise,export_measurement,load_cif_as_preset,find_preset,default_shape_for_material,
    compatible_shapes,cells_for_size_nm,to_frame,offset_report)
    from . import facets
else:                                # run from a plain folder
    from bcdi_core import (MATERIAL_PRESETS,DETECTOR_PRESETS,BCDIConfig,CrystalBuilder,ReflectionCalculator,
    BCDISimulator,DislocationConfig,DislocationLoopConfig,lattice_html_with_controls,bragg_html_with_controls,
    add_experimental_noise,export_measurement,load_cif_as_preset,find_preset,default_shape_for_material,
    compatible_shapes,cells_for_size_nm,to_frame,offset_report)
    import facets

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
class _Sci(QLineEdit):
    """Number field that accepts 1e9, 1.5e12 or a plain number."""
    def __init__(self,v=1e9,w=110):
        super().__init__(f'{v:g}');self.setMaximumWidth(w)
    def value(self):
        try: return float(self.text().strip().replace(' ',''))
        except ValueError: return 0.0
    def setValue(self,v): self.setText(f'{float(v):g}')

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
        split=QSplitter(Qt.Orientation.Horizontal)
        sc=QScrollArea();sc.setWidgetResizable(True);sc.setMinimumWidth(340);inner=QWidget();ll=QVBoxLayout(inner);ll.setSpacing(6);ll.setContentsMargins(4,4,8,4)
        h=QHBoxLayout();t=QLabel("Material & Lattice");t.setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700");h.addWidget(t,1);self.nb=_nb();h.addWidget(self.nb);ll.addLayout(h)
        # Material with CIF import
        mb=QGroupBox("Material");mv=QVBoxLayout(mb);mv.setSpacing(4)
        r1=QHBoxLayout();r1.addWidget(QLabel("Preset:"));self.combo=QComboBox()
        for k in MATERIAL_PRESETS: self.combo.addItem(k)
        self.combo.currentTextChanged.connect(self._pc);r1.addWidget(self.combo,1);mv.addLayout(r1)
        r2=QHBoxLayout();r2.addWidget(QLabel("Formula:"));self.fe=QLineEdit();self.fe.setPlaceholderText("e.g. Fe5GeTe2");r2.addWidget(self.fe,1)
        mb2=QPushButton("Match");mb2.setStyleSheet("background:#4f98a3;padding:5px 10px;min-height:20px");mb2.setMaximumWidth(55);mb2.clicked.connect(self._mt);r2.addWidget(mb2)
        cif_btn=QPushButton("Load .cif");cif_btn.setStyleSheet("background:#1f6feb;padding:5px 10px;min-height:20px");cif_btn.setMaximumWidth(70);cif_btn.clicked.connect(self._load_cif);r2.addWidget(cif_btn)
        mv.addLayout(r2);self.il=QLabel();self.il.setWordWrap(True);self.il.setStyleSheet("color:#8b949e;font-size:9pt");mv.addWidget(self.il);ll.addWidget(mb)
        # Supercell
        sb=QGroupBox("Supercell");sf=QFormLayout(sb);sf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);sf.setVerticalSpacing(4)
        row=QHBoxLayout();self.nx=QSpinBox();self.ny=QSpinBox();self.nz=QSpinBox()
        for s in(self.nx,self.ny,self.nz): s.setRange(1,100000);s.setValue(20);s.setMinimumWidth(55);s.valueChanged.connect(self._us)
        row.addWidget(QLabel("nx"));row.addWidget(self.nx);row.addWidget(QLabel("ny"));row.addWidget(self.ny);row.addWidget(QLabel("nz"));row.addWidget(self.nz);row.addStretch()
        sf.addRow("Cells:",row)
        nmr=QHBoxLayout();self.size_nm=_dbl(0.5,20000,8,1," nm");nmr.addWidget(self.size_nm)
        bn=QPushButton("set cells");bn.setMaximumWidth(80);bn.setStyleSheet("background:#4f98a3;padding:4px 8px;min-height:20px")
        bn.clicked.connect(self._from_nm);nmr.addWidget(bn);nmr.addStretch();sf.addRow("Size:",nmr)
        self.shp=QComboBox();self.shp.currentTextChanged.connect(self._shape_changed);sf.addRow("Shape:",self.shp)
        fr=QHBoxLayout()
        self.fac_on=QCheckBox();self.wulff=_dbl(0.5,2.5,1.15,2);self.wulff.setEnabled(False)
        self.fac_lbl=QLabel("Facet multiplicities:")
        for w in (self.fac_on,self.wulff,self.fac_lbl): w.setToolTip('Controls the balance between the flat faces (facets) of the crystal.\\n\\nReal crystals grow flat faces whose relative size depends on which face is\\nmost stable. This number is the distance to the square {100} faces divided\\nby the distance to the triangular {111} faces.\\n\\n   below 0.58   pure cube, only square faces\\n   around 1.15  truncated octahedron, the usual shape of FCC metals\\n                (aluminium, copper, gold)\\n   above 1.73   pure octahedron, only triangular faces\\n\\nIn between you get a mix: raising the number grows the triangular faces\\nand shrinks the square ones. Leave the box unticked to use the standard 1.15.')
        self.fac_on.toggled.connect(self.wulff.setEnabled)
        fr.addWidget(self.fac_on);fr.addWidget(self.wulff);fr.addStretch()
        sf.addRow(self.fac_lbl,fr);self.sl=QLabel();self.sl.setStyleSheet("color:#8b949e;font-size:9pt");sf.addRow(self.sl);ll.addWidget(sb)
        vb=QGroupBox("Visualization");vv=QHBoxLayout(vb);vv.setSpacing(10)
        self.bc=QCheckBox("Bonds");self.bc.setToolTip("Draw a line between neighbouring atoms. Clear for small cells, slow above ~80k atoms.")
        self.stc=QCheckBox("Strain");self.stc.setToolTip("Colour every atom by how far it sits from its ideal lattice site (inferno scale).")
        self.dvc=QCheckBox("Dislocation");self.dvc.setToolTip("Dim the undisturbed lattice and colour only the displaced atoms, so the defect stands out.")
        for w in (self.bc,self.stc,self.dvc): vv.addWidget(w)
        vv.addStretch();ll.addWidget(vb)
        # Dislocation: line + loop tabs
        db=QGroupBox("Defects");dv=QVBoxLayout(db);dv.setSpacing(4)
        # Line dislocation
        self.dck=QCheckBox("Line dislocation");dv.addWidget(self.dck)
        self.df=QFrame();dg=QGridLayout(self.df);dg.setContentsMargins(2,2,2,2);dg.setHorizontalSpacing(8);dg.setVerticalSpacing(5)
        self.dt=QComboBox();self.dt.addItems(["edge","screw","mixed"]);self.dd=QComboBox();self.dd.addItems(["Z","Y","X"])
        dg.addWidget(QLabel("Type:"),0,0,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dt,0,1);dg.addWidget(QLabel("Line:"),0,2,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dd,0,3)
        self.dpx=_dbl(0,1,.5,3);self.dpy=_dbl(0,1,.5,3)
        dg.addWidget(QLabel("X:"),1,0,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dpx,1,1);dg.addWidget(QLabel("Y:"),1,2,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dpy,1,3)
        self.db_=_dbl(0,50,0,3," A");self.db_.setSpecialValueText("auto");self.dnu=_dbl(0,.5,.3,3)
        dg.addWidget(QLabel("|b|:"),2,0,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.db_,2,1);dg.addWidget(QLabel("nu:"),2,2,Qt.AlignmentFlag.AlignRight);dg.addWidget(self.dnu,2,3)
        self.df.setVisible(False);dv.addWidget(self.df);self.dck.toggled.connect(self.df.setVisible)
        # Loop dislocation
        self.lck=QCheckBox("Dislocation loop (prismatic)");dv.addWidget(self.lck)
        self.lf=QFrame();lg=QGridLayout(self.lf);lg.setContentsMargins(2,2,2,2);lg.setHorizontalSpacing(8);lg.setVerticalSpacing(5)
        self.lcx=_dbl(0,1,.5,3);self.lcy=_dbl(0,1,.5,3);self.lcz=_dbl(0,1,.5,3)
        lg.addWidget(QLabel("Cx:"),0,0,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lcx,0,1);lg.addWidget(QLabel("Cy:"),0,2,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lcy,0,3);lg.addWidget(QLabel("Cz:"),0,4,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lcz,0,5)
        self.lr=_dbl(1,500,20,1," A");self.lb=_dbl(0,50,0,3," A");self.lb.setSpecialValueText("auto");self.ln=QComboBox();self.ln.addItems(["Z","Y","X"]);self.lnu2=_dbl(0,.5,.3,3)
        lg.addWidget(QLabel("R:"),1,0,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lr,1,1);lg.addWidget(QLabel("|b|:"),1,2,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.lb,1,3);lg.addWidget(QLabel("N:"),1,4,Qt.AlignmentFlag.AlignRight);lg.addWidget(self.ln,1,5)
        self.lf.setVisible(False);dv.addWidget(self.lf);self.lck.toggled.connect(self.lf.setVisible)
        ll.addWidget(db)
        stb=QGroupBox("Analytical strain");stf=QFormLayout(stb);stf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);stf.setVerticalSpacing(4)
        self.strc=QComboBox();self.strc.addItems(["none","radial_gradient","edge_dislocation","random"]);stf.addRow("Type:",self.strc)
        self.strm=_dbl(0,.1,1e-4,6);stf.addRow("eps:",self.strm);ll.addWidget(stb)
        self.bb=QPushButton("Build & Visualize");self.bb.setStyleSheet("background:#1f6feb;min-height:34px;font-size:11pt");self.bb.clicked.connect(self._bd);ll.addWidget(self.bb)
        self.pg=QProgressBar();ll.addWidget(self.pg);self.lg=QPlainTextEdit();self.lg.setReadOnly(True);self.lg.setMaximumHeight(65);ll.addWidget(self.lg);ll.addStretch();sc.setWidget(inner);split.addWidget(sc)
        right=QWidget();rv=QVBoxLayout(right);rv.setContentsMargins(0,0,0,0);rv.addWidget(QLabel("3D Lattice"))
        if _HAS_WEB: self.viz=QWebEngineView();self.viz.setHtml(_ph("Build lattice."))
        else: self.viz=QLabel("Need PyQt6-WebEngine")
        self.viz.setMinimumHeight(400);rv.addWidget(self.viz,1);split.addWidget(right)
        split.setStretchFactor(0,0);split.setStretchFactor(1,1);split.setSizes([480,1000]);root.addWidget(split)
        self._pc(self.combo.currentText());self._us()
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
    def _from_nm(self):
        try:
            n=cells_for_size_nm(self.combo.currentText(),self.size_nm.value())
            for w,v in zip((self.nx,self.ny,self.nz),n): w.blockSignals(True);w.setValue(v);w.blockSignals(False)
            self._us()
        except Exception as e: self.lg.appendPlainText(f"size: {e}")
    def _shape_changed(self,t):
        on=(t=='wulff')
        for w in (self.fac_on,self.fac_lbl): w.setEnabled(on)
        self.wulff.setEnabled(on and self.fac_on.isChecked())
    def _us(self):
        try:
            cfg=BCDIConfig(self.combo.currentText());cfg.SUPERCELL_MULT=(self.nx.value(),self.ny.value(),self.nz.value())
            sz=cfg.particle_size_nm;nc=self.nx.value()*self.ny.value()*self.nz.value();na=nc*len(cfg.BASIS_COORDS)
            f=lambda v:f"{v:.1f}nm" if v<100 else f"{v/1000:.3f}um"
            w="" if na<2_000_000 else ("   [heavy: minutes + GBs on most machines]" if na<2e7 else "   [very heavy: expect long build times]")
            self.sl.setText(f"-> {f(sz[0])} x {f(sz[1])} x {f(sz[2])}  {nc:,} cells  ~{na:,} atoms{w}")
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
        cfg.WULFF_RATIO=self.wulff.value() if self.fac_on.isChecked() else 1.15
        na=cfg.SUPERCELL_MULT[0]*cfg.SUPERCELL_MULT[1]*cfg.SUPERCELL_MULT[2]*len(cfg.BASIS_COORDS)
        if na>10_000_000 and QMessageBox.question(self,"Large supercell",
                f"~{na:,} atoms.\nBuild time and memory scale with this and depend on your machine.\nContinue?"
                )!=QMessageBox.StandardButton.Yes: return
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
        self.keV=_dbl(.5,100,10,2," keV");self.keV.valueChanged.connect(self._ul);self.keV.valueChanged.connect(lambda:self._rt.start())
        self.ll_=QLabel();self.ll_.setStyleSheet("color:#8b949e;font-size:9pt");self._ul();self.pol=_dbl(0,1,1,2);self.bsz=_dbl(0.1,1000,1,1," um")
        self.nph=_Sci(1e9);self.nph.setToolTip("Photons in the simulated exposure. Type it as 1e9, 2.5e12 or a plain number.\n"
            "Only applied when Poisson noise is ticked in the Results tab.")
        self.phmode=QComboBox();self.phmode.addItems(["total","peak"])
        bf.addRow("Energy:",self.keV);bf.addRow("",self.ll_);bf.addRow("Pol:",self.pol);bf.addRow("Beam size:",self.bsz)
        bf.addRow("Photons:",self.nph);bf.addRow("counted as:",self.phmode);lc.addWidget(bb)
        # Detector with preset selector and NX x NY
        db=QGroupBox("Detector");dv=QVBoxLayout(db);dv.setSpacing(3)
        pr=QHBoxLayout();pr.addWidget(QLabel("Preset:"));self.det_combo=QComboBox()
        for k in DETECTOR_PRESETS: self.det_combo.addItem(k)
        self.det_combo.currentTextChanged.connect(self._det_preset);pr.addWidget(self.det_combo,1);dv.addLayout(pr)
        df=QFormLayout();df.setLabelAlignment(Qt.AlignmentFlag.AlignRight);df.setVerticalSpacing(3)
        self.det_d=_dbl(.05,10,.5,3," m")
        # NX x NY row
        pxr=QHBoxLayout();self.det_nx=QSpinBox();self.det_nx.setRange(32,4096);self.det_nx.setValue(256)
        pxr.addWidget(self.det_nx);pxr.addWidget(QLabel("x"));self.det_ny=QSpinBox();self.det_ny.setRange(32,4096);self.det_ny.setValue(256);pxr.addWidget(self.det_ny)
        for _w in (self.det_nx,self.det_ny):
            _w.setToolTip("Detector panel in pixels. Editing it by hand also sets the image size below.")
            _w.valueChanged.connect(self._panel_edited)
        self.det_sz=_dbl(1,500,55,1," um")
        df.addRow("Distance:",self.det_d);df.addRow("Pixels:",pxr);df.addRow("Pixel size:",self.det_sz)
        # Rocking: "Steps" label (= number of rocking curve angular positions)
        self.rock=_dbl(.0001,1,.00315,5," deg");self.rock.setDecimals(5)
        self.steps=QSpinBox();self.steps.setRange(8,2048);self.steps.setValue(128)
        df.addRow("Rock step:",self.rock);df.addRow("Steps:",self.steps)
        # Oversampling with tooltip
        self.os=_dbl(1,20,5,1);self.os.setToolTip("Oversampling ratio: simulation grid = OS x Nyquist.\nHigher = finer q-resolution but slower.\nTypical BCDI: 3-8")
        os_row=QHBoxLayout();os_row.addWidget(self.os);os_lbl=QLabel("(oversampling)");os_lbl.setStyleSheet("color:#8b949e;font-size:8pt");os_row.addWidget(os_lbl)
        df.addRow("OS ratio:",os_row)
        self.grid=QSpinBox();self.grid.setRange(32,512);self.grid.setValue(256)
        self.grid.setToolTip("Pixels across the simulated image. More pixels sample each fringe more\n"
            "finely, so the 2D slices get smoother with real data, not interpolation.\n"
            "Oversampling follows automatically; override it below if you want.\n"
            "Cost grows as N^3: 128 ~0.5 s, 256 ~5 s and ~1 GB, 384 ~16 s and ~3.6 GB.")
        self.grid.valueChanged.connect(self._grid_changed)
        df.addRow("Image size (px):",self.grid)
        dv.addLayout(df);lc.addWidget(db)
        # Coherence
        cb=QGroupBox("Coherence");cf=QFormLayout(cb);cf.setLabelAlignment(Qt.AlignmentFlag.AlignRight);cf.setVerticalSpacing(3)
        self.sh=_dbl(0,10000,300,1," um");self.sv=_dbl(0,10000,10,1," um");self.sd=_dbl(.1,500,50,1," m");self.bw=_dbl(1e-6,.1,1e-4,6);self.bw.setDecimals(6)
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
        ar=QHBoxLayout();ar.addWidget(QLabel("Custom 2th:"))
        self.custom_angle=_dbl(0,180,0,3," deg");self.custom_angle.setSpecialValueText("auto")
        self.custom_angle.setToolTip("Park the detector away from the exact Bragg angle.\n"
            "The peak moves off centre and the recorded intensity falls, exactly as it does\n"
            "when you step off a rocking curve. Leave at 0 to sit on the Bragg condition.")
        self.custom_angle.valueChanged.connect(self._ang_info);ar.addWidget(self.custom_angle,1);rc.addLayout(ar)
        self.ang_lbl=QLabel("");self.ang_lbl.setWordWrap(True);self.ang_lbl.setStyleSheet("color:#8b949e;font-size:9pt");rc.addWidget(self.ang_lbl)
        mh.addLayout(rc,1);root.addLayout(mh,1)
        self.sb_=QPushButton("Simulate");self.sb_.setStyleSheet("background:#1f6feb;min-height:34px;font-size:11pt");self.sb_.setEnabled(False);self.sb_.clicked.connect(self._rn);root.addWidget(self.sb_)
        self.pg=QProgressBar();root.addWidget(self.pg);self.lg=QPlainTextEdit();self.lg.setReadOnly(True);self.lg.setMaximumHeight(55);root.addWidget(self.lg)
        self._grid_changed(self.grid.value())
    def _det_preset(self,name):
        """A preset describes the physical panel; it must not blow up the FFT grid."""
        if name in DETECTOR_PRESETS and name!='Custom':
            d=DETECTOR_PRESETS[name];self._lock=True
            self.det_nx.setValue(d['nx']);self.det_ny.setValue(d['ny']);self.det_sz.setValue(d['pixel_um'])
            self._lock=False

    def _panel_edited(self):
        if getattr(self,'_lock',False): return
        self.grid.setValue(min(max(self.det_nx.value(),self.det_ny.value()),self.grid.maximum()))

    def _grid_changed(self,v):
        # keep the displayed q-window fixed: finer pixels mean finer sampling
        self.os.setValue(round(v/25.6,1))
        if hasattr(self,'ang_lbl'): self._ang_info()
    def _ang_info(self):
        t=self.custom_angle.value()
        if self.config is None or t<=0.01 or self._rdf is None or not len(self._rdf):
            self.ang_lbl.setText("");return
        near=self._rdf.iloc[(self._rdf['2theta']-t).abs().argsort().iloc[0]]
        off=t-near['2theta']
        c=self.config;c.TARGET_OVERSAMPLING=self.os.value();c.DETECTOR_PIXEL_UM=self.det_sz.value()
        c.SAMPLE_DETECTOR_DISTANCE_M=self.det_d.value();c.SIM_GRID_MAX=self.grid.value()
        c.DETECTOR_NX=self.det_nx.value();c.DETECTOR_NY=self.det_ny.value()
        dq,vox,pix,gone=offset_report(c,np.radians(near['theta_B']),off)
        msg=(f"nearest {near['hkl_display']} at {near['2theta']:.3f} deg -> offset {off:+.3f} deg\n"
             f"dq = {dq:+.4f} 1/A : peak moves {vox:+.1f} voxels ({pix:+.0f} real pixels)")
        self.ang_lbl.setText(msg+("\n  peak leaves the simulated frame" if gone else ""))
        self.ang_lbl.setStyleSheet("color:#f0883e;font-size:9pt" if gone else "color:#8b949e;font-size:9pt")

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
            self._dc(df);self._ang_info()
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
        c=self.config;c.BEAM_ENERGY_KEV=self.keV.value();c.POLARIZATION_FACTOR=self.pol.value();c.BEAM_SIZE_UM=self.bsz.value();c.N_PHOTONS=self.nph.value();c.PHOTON_MODE=self.phmode.currentText()
        c.DET_COUNTING=bool(DETECTOR_PRESETS.get(self.det_combo.currentText(),{}).get('counting',True))
        c.SAMPLE_DETECTOR_DISTANCE_M=self.det_d.value();c.DETECTOR_NX=self.det_nx.value();c.DETECTOR_NY=self.det_ny.value();c.SIM_GRID_MAX=self.grid.value()
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
        if c.SIM_GRID_MAX>320 and QMessageBox.question(self,"Large image",
                f"{c.SIM_GRID_MAX} px needs roughly {(c.SIM_GRID_MAX/384)**3*16:.0f} s and "
                f"{(c.SIM_GRID_MAX/384)**3*3.6:.1f} GB on a typical machine.\nContinue?"
                )!=QMessageBox.StandardButton.Yes:
            self.sb_.setEnabled(True);return
        self.sb_.setEnabled(False);self.nb.setEnabled(False);self.pg.setValue(0);self.sim_req.emit({'config':c,'builder':self.builder,'hkl':hkl})

# ======================= TAB 3 =======================
class T3(QWidget):
    def __init__(self): super().__init__();self._s=self._r=self._c=None;self._ui()
    def _ui(self):
        root=QVBoxLayout(self);root.setContentsMargins(12,12,12,12);root.setSpacing(6)
        h=QHBoxLayout();h.addWidget(QLabel("Diffraction Results"));h.itemAt(0).widget().setStyleSheet("color:#4f98a3;font-size:13pt;font-weight:700")
        h.addSpacing(15);h.addWidget(QLabel("Effects:"));h.itemAt(2).widget().setStyleSheet("color:#8b949e;font-size:9pt")
        self.np_=QCheckBox("Poisson")
        self.np_.setToolTip("Counting statistics for the photon budget set in the Beam tab.\n"
            "Leave off to see the ideal pattern: weak fringes survive only at high photon counts.")
        self.nr=QCheckBox("Readout");self.nr.setToolTip("Integrating (CCD/CMOS) detectors only. Photon-counting "
            "detectors (EIGER, PILATUS, MAXIPIX, MERLIN, TIMEPIX) have no read noise.")
        self.na=QCheckBox("Air");self.nd=QCheckBox("Dead px");self.nh=QCheckBox("Hot px")
        for c in(self.np_,self.nr,self.na,self.nd,self.nh): h.addWidget(c)
        h.addSpacing(12);h.addWidget(QLabel("Frame:"));h.itemAt(h.count()-1).widget().setStyleSheet("color:#8b949e;font-size:9pt")
        self.frame=QComboBox();self.frame.addItems(["sample (lab, orthogonal q)","detector (raw, skewed)"])
        self.frame.setMaximumWidth(190);self.frame.currentTextChanged.connect(lambda _:self._rr());h.addWidget(self.frame)
        self.rb=QPushButton("Re-render");self.rb.setEnabled(False);self.rb.setStyleSheet("background:#4f98a3;padding:4px 10px;min-height:20px;font-size:9pt");self.rb.clicked.connect(self._rr);h.addWidget(self.rb)
        h.addStretch();root.addLayout(h)
        self.il=QLabel("Run simulation.");self.il.setStyleSheet("color:#8b949e;font-size:9pt");root.addWidget(self.il)
        sp=QSplitter(Qt.Orientation.Horizontal);lp=QWidget();ll=QVBoxLayout(lp);ll.setContentsMargins(0,0,0,0);ll.addWidget(QLabel("3D Bragg Peak (measure: check box in controls)"))
        if _HAS_WEB: self.bv=QWebEngineView();self.bv.setHtml(_ph("3D after sim."))
        else: self.bv=QLabel("No WebEngine")
        self.bv.setMinimumWidth(520);ll.addWidget(self.bv,1);sp.addWidget(lp)
        rp=QWidget();rl=QVBoxLayout(rp);rl.setContentsMargins(0,0,0,0);rl.setSpacing(4);rl.addWidget(QLabel("2D Slices"))
        with matplotlib.rc_context(MPL_DARK): self.f1=Figure(figsize=(5,3.4),dpi=200,tight_layout={'pad':0.4});self.f2=Figure(figsize=(5,3.4),dpi=200,tight_layout={'pad':0.4})
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
        """None when nothing is ticked, so the ideal pattern is shown untouched."""
        c=self._c;o={}
        if self.np_.isChecked():
            o['poisson']=True
            o['n_photons']=getattr(c,'N_PHOTONS',0) if c is not None else 0
            o['photon_mode']=getattr(c,'PHOTON_MODE','total') if c is not None else 'total'
        if self.nr.isChecked(): o['readout_noise']=5.0
        if self.na.isChecked(): o['air_scatter']=100.0
        if self.nd.isChecked(): o['dead_pixels_frac']=0.002
        if self.nh.isChecked(): o['hot_pixels_frac']=0.0005
        if not o: return None
        o.setdefault('poisson',False);o['counting']=bool(getattr(c,'DET_COUNTING',True))
        return o
    def _rr(self):
        if self._s: self._di(self._s,self._r,self._c,self._gn())
    def _export(self):
        if self._s is None: return
        path,_=QFileDialog.getSaveFileName(self,"Export measurement","bcdi_measurement","NumPy (*.npz);;HDF5 (*.h5)")
        if not path: return
        try:
            self._c.Q_FRAME='det' if self.frame.currentIndex()==1 else 'lab'
            export_measurement(self._s,self._r,self._c,path,self._gn());QMessageBox.information(self,"Exported",f"Saved to {path}")
        except Exception as e: QMessageBox.critical(self,"Export Error",str(e))
    def display(self,sim,refl,cfg): self._s=sim;self._r=refl;self._c=cfg;self.rb.setEnabled(True);self.exp_btn.setEnabled(True);self._di(sim,refl,cfg,self._gn())
    def _di(self,sim,refl,cfg,no):
        fl='detector frame (raw, skewed)' if self.frame.currentIndex()==1 else 'sample / lab frame (orthogonal q)'
        self.nr.setEnabled(not bool(getattr(cfg,'DET_COUNTING',True)))
        if not self.nr.isEnabled(): self.nr.setChecked(False)
        self.il.setText(f"{cfg.MATERIAL_NAME} {refl['hkl_str']} lam={cfg.wavelength_angstrom:.3f}A "
            f"2th={np.degrees(refl['two_theta_rad']):.3f} det={cfg.DETECTOR_NX}x{cfg.DETECTOR_NY} grid={cfg.DETECTOR_N_PIXELS}^3 "
            f"photons={getattr(cfg,'N_PHOTONS',0):.3g} | {fl}")
        vol=sim.diff_volume.copy()
        if no: vol=add_experimental_noise(vol,**no)
        vol,_=to_frame(sim,'det' if self.frame.currentIndex()==1 else 'lab',vol)
        if _HAS_WEB:
            try:
                html=bragg_html_with_controls(sim,2,1e-4,None,vol=vol,frame_label=fl)
                if html: f=tempfile.NamedTemporaryFile('w',suffix='.html',delete=False,encoding='utf-8');f.write(html);f.close();self.bv.load(QUrl.fromLocalFile(f.name))
            except Exception as e: print(f"3D: {e}")
        qx=sim.q_grids['qx'];qy=sim.q_grids['qy'];qz=sim.q_grids['qz'];N=vol.shape[0];cn=N//2;lmx=np.log10(vol.max()+1.);lmn=max(0,lmx-6)
        for fig,title,data,qh,qv,xl,yl in [(self.f1,"qx-qy",vol[:,:,cn],qx,qy,"qx","qy"),(self.f2,"qx-qz",vol[:,cn,:],qx,qz,"qx","qz")]:
            with matplotlib.rc_context(MPL_DARK):
                fig.clear();ax=fig.add_subplot(111);ax.set_facecolor('#000005');ld=np.log10(data.T+1.)
                im=ax.imshow(ld,origin='lower',cmap='jet',aspect='auto',extent=[qh[0],qh[-1],qv[0],qv[-1]],vmin=lmn,vmax=lmx,interpolation='nearest')
                tA=cfg.particle_size_angstrom;dq=2*np.pi/np.mean(tA[:2]);qz_=8*dq;ax.set_xlim(-qz_,qz_);ax.set_ylim(-qz_,qz_)
                div=make_axes_locatable(ax);cax=div.append_axes('right',size='2.5%',pad=.035)
                cb=fig.colorbar(im,cax=cax);cb.set_label('intensity arb.',fontsize=6);cb.ax.tick_params(labelsize=5)
                ax.set_title(f'{title}',fontsize=7)
                ax.set_xlabel(xl+' (1/A)',fontsize=6);ax.set_ylabel(yl+' (1/A)',fontsize=6);ax.tick_params(labelsize=5)
        self.c1.draw();self.c2.draw()

# ======================= MAIN =======================
class MW(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle("BCDI Simulator v6.0");self.resize(1500,960);self.setMinimumSize(1200,800)
        self.tabs=QTabWidget();self.tabs.setDocumentMode(True);self.t1=T1();self.t2=T2();self.t3=T3()
        self.tabs.addTab(self.t1," 1 Material ");self.tabs.addTab(self.t2," 2 Beam ");self.tabs.addTab(self.t3," 3 Results ")
        self.tabs.setTabEnabled(1,False);self.tabs.setTabEnabled(2,False);self.setCentralWidget(self.tabs)
        sb=QStatusBar();self.setStatusBar(sb);sb.showMessage("Build lattice first.")
        self.t1.built.connect(self._ob);self.t1.nb.clicked.connect(lambda:self.tabs.setCurrentIndex(1))
        self.t2.sim_req.connect(self._os);self.t2.nb.clicked.connect(lambda:self.tabs.setCurrentIndex(2));self._sw=None
    def _ob(self,c,b): self.t2.sc(c,b);self.tabs.setTabEnabled(1,True)
    def _os(self,p):
        self._sw=SW(p['config'],p['builder'],p['hkl']);self._sw.progress.connect(self.t2.pg.setValue);self._sw.log.connect(self.t2.lg.appendPlainText)
        self._sw.done.connect(self._sd);self._sw.failed.connect(self._sf);self._sw.start()
    def _sd(self,s,r,d): self.t2.sb_.setEnabled(True);self.t2.nb.setEnabled(True);self.t3.display(s,r,s.config);self.tabs.setTabEnabled(2,True)
    def _sf(self,m): self.t2.sb_.setEnabled(True);self.t2.lg.appendPlainText(f"FAIL: {m}")
def main():
    app=QApplication(sys.argv);app.setStyle("Fusion");pal=QPalette()
    for r,c in[(QPalette.ColorRole.Window,"#0d1117"),(QPalette.ColorRole.WindowText,"#e6edf3"),(QPalette.ColorRole.Base,"#0d1117"),(QPalette.ColorRole.AlternateBase,"#161b22"),(QPalette.ColorRole.Text,"#e6edf3"),(QPalette.ColorRole.Button,"#21262d"),(QPalette.ColorRole.ButtonText,"#e6edf3"),(QPalette.ColorRole.Highlight,"#1f6feb"),(QPalette.ColorRole.HighlightedText,"#ffffff")]:
        pal.setColor(r,QColor(c))
    app.setPalette(pal);app.setStyleSheet(QSS);win=MW();win.show();sys.exit(app.exec())
if __name__=="__main__": main()
