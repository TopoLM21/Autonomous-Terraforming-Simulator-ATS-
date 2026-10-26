"""Small editor for a new regional run; existing earthworks keep their material."""
from copy import deepcopy
from PySide6 import QtWidgets
from .soil import SoilProfile


class SoilDialog(QtWidgets.QDialog):
    def __init__(self,profile,parent=None,editable=True):
        super().__init__(parent);self.setWindowTitle('Условные слои грунта');self.resize(780,330)
        self.profile=deepcopy(profile or SoilProfile.qattara());self.result_profile=profile
        layout=QtWidgets.QVBoxLayout(self)
        note=QtWidgets.QLabel('Приближение для сценария Каттары, без геологической съёмки. Глубины отсчитываются от исходной поверхности. Коэффициент копания увеличивает время и энергию; 1 — рыхлый песок. Нижний материал — выветрелая порода, доступная ковшу, а не монолитная скала.')
        note.setWordWrap(True);layout.addWidget(note)
        self.enabled=QtWidgets.QCheckBox('Использовать условные слои');self.enabled.setChecked(profile is not None)
        layout.addWidget(self.enabled)
        self.settings=QtWidgets.QWidget();grid=QtWidgets.QGridLayout(self.settings)
        for col,title in enumerate(('Материал','Плотность, кг/м³','Коэффициент копания','Толщина, м')):
            grid.addWidget(QtWidgets.QLabel(title),0,col)
        self.densities=[];self.factors=[];self.thicknesses=[]
        for i,m in enumerate(self.profile.materials):
            grid.addWidget(QtWidgets.QLabel(m.name),i+1,0)
            density=QtWidgets.QDoubleSpinBox();density.setRange(100,5000);density.setDecimals(1);density.setSingleStep(50);density.setValue(m.density)
            factor=QtWidgets.QDoubleSpinBox();factor.setRange(1,100);factor.setSingleStep(.1);factor.setValue(m.digging_factor)
            self.densities.append(density);self.factors.append(factor)
            grid.addWidget(density,i+1,1);grid.addWidget(factor,i+1,2)
            if i<len(self.profile.thicknesses):
                depth=QtWidgets.QDoubleSpinBox();depth.setRange(.01,10000);depth.setSingleStep(.25);depth.setValue(self.profile.thicknesses[i]);self.thicknesses.append(depth)
                grid.addWidget(depth,i+1,3)
            else:grid.addWidget(QtWidgets.QLabel('Вся оставшаяся глубина'),i+1,3)
        layout.addWidget(self.settings)
        self.enabled.setEnabled(editable);self.settings.setEnabled(editable and self.enabled.isChecked())
        self.enabled.toggled.connect(lambda enabled:self.settings.setEnabled(editable and enabled))
        if not editable:
            locked=QtWidgets.QLabel('Профиль этого ландшафта уже используется. Его сохраняем вместе с грунтом; изменять свойства можно до открытия нового рабочего участка.')
            locked.setWordWrap(True);layout.addWidget(locked)
        buttons=QtWidgets.QDialogButtonBox((QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel) if editable else QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.apply);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def apply(self):
        profile=deepcopy(self.profile)
        for m,d,f in zip(profile.materials,self.densities,self.factors):m.density=d.value();m.digging_factor=f.value()
        profile.thicknesses=[s.value() for s in self.thicknesses];profile.validate()
        self.result_profile=profile if self.enabled.isChecked() else None;self.accept()
