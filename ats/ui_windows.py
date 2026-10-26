"""Compact workspace: the map stays visible, detailed controls open on demand."""
from PySide6 import QtCore,QtWidgets


def fit_window(window,width=1000,height=700):
    screen=window.screen() or QtWidgets.QApplication.primaryScreen()
    rect=screen.availableGeometry()
    window.resize(min(width,max(480,rect.width()-32)),min(height,max(360,rect.height()-64)))


def show_window(window):
    window.show();window.raise_();window.activateWindow()


def tool_window(host,title,widgets,width=900,height=600):
    dialog=QtWidgets.QDialog(host);dialog.setWindowTitle(title)
    layout=QtWidgets.QVBoxLayout(dialog)
    for widget in widgets:layout.addWidget(widget)
    fit_window(dialog,width,height);return dialog


def compact_workspace(host,split,right,right_layout,telemetry_scroll,settings_scroll,groups,profile_controls):
    host.resize(1180,760)
    title=host.windowTitle();host.setWindowTitle(title+' · карта')
    host.settings_window=QtWidgets.QDialog(host);host.settings_window.setWindowTitle('Настройки сценария и базы')
    layout=QtWidgets.QVBoxLayout(host.settings_window);host.settings_tabs=QtWidgets.QTabWidget();layout.addWidget(host.settings_tabs)
    for name,group in groups:
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(group)
        host.settings_tabs.addTab(scroll,name)
    fit_window(host.settings_window,580,680)
    right_layout.removeWidget(settings_scroll);settings_scroll.hide()
    right_layout.removeWidget(telemetry_scroll)
    host.telemetry_window=tool_window(host,'Подробная телеметрия',[telemetry_scroll],630,680)
    telemetry_scroll.show()
    right_layout.removeWidget(host.log)
    host.events_window=tool_window(host,'Журнал событий',[host.log],850,570);host.log.show()
    host.profile_window=tool_window(host,'Профили рельефа',[profile_controls,host.profile],1000,560)
    host.profile.show();profile_controls.show()
    host.fleet_window=tool_window(host,'Состояние дронов',[host.fleet_table],1100,500)
    host.fleet_table.setMaximumHeight(16777215);host.fleet_table.show()
    host.plan_window=tool_window(host,'Кисть и план',[host.plan_editor],520,680);host.plan_editor.show()
    # Remove the former journal heading; all other existing widget references
    # are retained so hot edits and saved UI state keep working.
    for i in reversed(range(right_layout.count())):
        item=right_layout.itemAt(i);widget=item.widget()
        if isinstance(widget,QtWidgets.QLabel) and widget.text()=='Журнал событий':
            right_layout.takeAt(i);widget.hide()
    host.compact_telemetry=QtWidgets.QLabel();host.compact_telemetry.setWordWrap(True)
    host.compact_telemetry.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    right_layout.addWidget(host.compact_telemetry)
    grid=QtWidgets.QGridLayout()
    def settings(index):
        host.settings_tabs.setCurrentIndex(index);show_window(host.settings_window)
    commands=[('Сценарий',lambda:settings(0)),('База и солнце',lambda:settings(1)),
        ('Энергоснабжение',host.show_power_design),('Комплексы · общая карта',host.show_complexes),
        ('Дроны',lambda:show_window(host.fleet_window)),('Профили',lambda:show_window(host.profile_window)),
        ('Телеметрия',lambda:show_window(host.telemetry_window)),('Журнал',lambda:show_window(host.events_window)),
        ('Паспорт и платформы',host.show_base_layout),
        ('Пыль и ветер',lambda:settings(2))]
    for i,(text,command) in enumerate(commands):
        button=QtWidgets.QPushButton(text);button.clicked.connect(command);grid.addWidget(button,i,0)
    right_layout.addLayout(grid);right_layout.addStretch()
    right.setMinimumWidth(205);right.setMaximumWidth(275)
    sidebar=QtWidgets.QScrollArea();sidebar.setWidgetResizable(True);sidebar.setWidget(right)
    sidebar.setMinimumWidth(225);sidebar.setMaximumWidth(295);sidebar.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    split.addWidget(sidebar);split.setSizes([920,230]);fit_window(host,1180,760)
    host.detail_windows=[host.settings_window,host.telemetry_window,host.events_window,host.profile_window,host.fleet_window,host.plan_window]


def menu_button(header,host,title):
    button=QtWidgets.QToolButton();button.setText(title);button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu=QtWidgets.QMenu(button);button.setMenu(menu);header.addWidget(button)
    return menu


def hidden_action(menu,host,button):
    button.setParent(host);button.hide();menu.addAction(button.text(),button.click)
