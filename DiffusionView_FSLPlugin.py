#!/usr/bin/env python
#
# A FSLeyes plugin for visualising diffusion MRI data
#
# S. Jbabdi 2022

import os.path as op

import numpy as np

import wx
import wx.html as wxhtml

import fsleyes_props as props

from   fsleyes.plotting.dataseries             import DataSeries, VoxelDataSeries
import fsleyes.profiles.timeseriesprofile      as     timeseriesprofile
import fsleyes.views.plotpanel                 as     plotpanel
import fsleyes.controls.plotcontrolpanel       as     plotcontrol
from   fsleyes.controls import plottoolbar
import fsleyes.controls.controlpanel           as     ctrlpanel

import fsleyes.actions as actions
import fsleyes.icons as icons
import fsleyes.tooltips as tooltips

from fsl.data.image import Image

class DTIFitter:
    """Calculate and cache DTI model fit at a single voxel"""
    def __init__(self, image, bvals, bvecs, add_kurtosis=False):

        self.image = image
        self.bvals = bvals
        self.bvecs = bvecs

        # rescale bvals from s/mm^2 --> ms/um^2
        if max(self.bvals) > 50:
            self.bvals *= 1e-3

        self.xyz     = None
        self.betas   = None
        self.tensor  = None
        self.eigvals = None
        self.eigvecs = None

        # kurtosis
        self.add_kurtosis = add_kurtosis

        # dti matrix
        self.dtimat    = self.form_DTImat(self.bvals, self.bvecs)
        self.dtimatinv = np.linalg.pinv(self.dtimat)


    @staticmethod
    def betas_to_tensor(betas):
        return [[betas[1], betas[2], betas[3]],
                [betas[2], betas[4], betas[5]],
                [betas[3], betas[5], betas[6]]]

    def form_DTImat(self, bval, bvec):
        """
        Create matrix for running GLM to fit DTI model
        Args:
        bval : (N,) array
        bvec : (N,3) array

        Returns:
        (N,7) array
        """
        desmat = np.array([np.ones_like(bvec[:,0]),
                         -bval * bvec[:, 0] * bvec[:, 0],
                         -bval * bvec[:, 0] * bvec[:, 1]*2.,
                         -bval * bvec[:, 0] * bvec[:, 2]*2.,
                         -bval * bvec[:, 1] * bvec[:, 1],
                         -bval * bvec[:, 1] * bvec[:, 2]*2.,
                         -bval * bvec[:, 2] * bvec[:, 2]
                         ]).T
        
        if self.add_kurtosis:            
            kurt_regressor = bval[:,None]**2/6.
            desmat         = np.concatenate([desmat,kurt_regressor],axis=1)

        return desmat        


    def calc_fit(self, x, y, z):
        """Run DTIFIT unless the voxel coordinates have not changed"""
        if self.xyz == (x, y, z):
            return
        data        = self.image[x, y, z, :]
        self.xyz    = (x, y, z)        
        
        self.betas  = np.dot(self.dtimatinv, np.log(np.abs(data)))
        self.tensor = DTIFitter.betas_to_tensor(self.betas)

        eigvals, eigvecs = np.linalg.eigh(self.tensor)
        self.eigvals = eigvals
        self.eigvecs = eigvecs


    def dtifit(self, x, y, z):
        """Return DTI model prediction """
        self.calc_fit(x, y, z)
        return np.exp(np.dot(self.dtimat, self.betas))

    def residuals(self, x, y, z):
        """Return model residuals"""
        self.calc_fit(x, y, z)
        data = self.image[x, y, z, :]
        pred = self.dtifit(x, y, z)
        return data-pred


    def fa(self, x, y, z):
        """Return Fractional Anisotropy"""
        md = self.md(x, y, z)
        l  = self.eigvals
        return np.sqrt(3./2.*np.sum((l-md)**2) / np.sum(l**2))

    def md(self, x, y, z):
        """Return mean diffusivity"""
        self.calc_fit(x, y, z)
        return sum(self.eigvals) / 3

    def v1(self, x, y, z):
        """Return principal eigenvector"""
        self.calc_fit(x, y ,z)
        return self.eigvecs[:, -1]   
        
    def kurt(self, x, y, z):
        """Return mean kurtosis"""
        if not self.add_kurtosis:
            return None
        self.calc_fit(x, y, z)
        
        return self.betas[7] / (self.md(x,y,z)**2)


# Helper functions

def set_dti_series(ds,bvals,bvecs,fitter):
    """Set display parameters for DTI series"""
    ds.colour    = "#f54242"
    ds.lineWidth = 1
    ds.lineStyle = '-'
    ds.alpha     = 1.0
    ds.label     = 'DTIfit'
    
    # hack to make DiffusionPlotView.prepareDataSeries work
    ds.fitter = fitter
    ds.bvals  = bvals
    ds.bvecs  = bvecs

def set_dki_series(ds,bvals,bvecs,fitter):
    """Set display parameters for DKI series (kurtosis version)"""
    ds.colour    = "#fcba03"
    ds.lineWidth = 1
    ds.lineStyle = '-'
    ds.alpha     = 1.0
    ds.label     = 'DKIfit'
    
    # hack to make DiffusionPlotView.prepareDataSeries work
    ds.fitter = fitter
    ds.bvals  = bvals
    ds.bvecs  = bvecs
   
def set_residual_series(ds,bvals,bvecs,fitter):
    """Set display parameters for the residual time series"""
    ds.colour    = "#0a0a0a"
    ds.lineWidth = 1
    ds.lineStyle = ':'
    ds.alpha     = 1.0
    ds.label     = 'Residuals'
    
    # hack to make DiffusionPlotView.prepareDataSeries work
    ds.fitter = fitter
    ds.bvals  = bvals
    ds.bvecs  = bvecs
   

def num_shells(bvals,tol=None):
    """Calculate the number of shells given some tolerance level
    """
    if tol is not None:
        return np.unique(np.round(bvals / tol)*tol, return_counts=True)
    else:
        return np.unique(bvals, return_counts=True)


def bvals_round(bvals, tol=0.1):
    """Returns rounded bvals"""
    tol = np.ceil(min(bvals[bvals!=0]))*tol
    return np.round(bvals / tol)*tol
 
 
def sort_with_bvals_and_v1(bvals, bvecs, v1):
    """Sort time series according to bvals then bvecs"""
    unique_bvals,_ = num_shells(bvals, tol = 100)
    idxs  = []
    for b in unique_bvals:
        bindxs = np.where(np.isclose(bvals, b, atol = 100))[0].tolist()
        if bvecs.shape[0]<bvecs.shape[1]:
            v = bvecs[:,bindxs].T
        else:
            v = bvecs[bindxs,:]
        dp = np.abs(np.dot(v1, v.T))
        idxs.extend( [bindxs[i] for i in np.argsort(dp)] ) 
        
    return idxs

class DiffusionDataSeries(VoxelDataSeries):

    def __init__(self, overlay, overlayList, displayCtx, plot):
        super().__init__(overlay, overlayList, displayCtx, plot.canvas)
        ovlDir = op.dirname(overlay.dataSource)

        self.difPanel = plot
        self.bvals = np.loadtxt(op.join(ovlDir, 'bvals'))
        self.bvecs = np.loadtxt(op.join(ovlDir, 'bvecs'))
        if self.bvecs.shape[0]<self.bvecs.shape[1]:
            self.bvecs = self.bvecs.T

        # info on shells
        s,n         = num_shells(self.bvals, tol=100)
        self.shells = dict(zip(s,n))

        self.fitter = DTIFitter(overlay, self.bvals, self.bvecs)
        
        self.dti_fitter = DTIFitter(overlay, self.bvals, self.bvecs)
        self.dtiseries = DataSeries(overlay, overlayList, displayCtx, plot.canvas)
        set_dti_series( self.dtiseries, self.bvals, self.bvecs, self.dti_fitter)
        self.dtiseries_res = DataSeries(overlay, overlayList, displayCtx, plot.canvas)
        set_residual_series( self.dtiseries_res, self.bvals, self.bvecs, self.dti_fitter)

        # kurtosis
        self.dki_fitter = DTIFitter(overlay, self.bvals, self.bvecs, add_kurtosis=True)       
        self.dkiseries = DataSeries(overlay, overlayList, displayCtx, plot.canvas)
        set_dki_series( self.dkiseries, self.bvals, self.bvecs, self.dki_fitter)
        self.dkiseries_res = DataSeries(overlay, overlayList, displayCtx, plot.canvas)
        set_residual_series( self.dkiseries_res, self.bvals, self.bvecs, self.dki_fitter)


    def extraSeries(self):
        """Add model prediction and residuals to the data to be plotted
        """
        outSeries = []
        if not self.difPanel.showDTIFit:
            return outSeries

        opts    = self.displayCtx.getOpts(self.overlay)
        x, y, z = opts.getVoxel()
        xdata   = np.arange(self.overlay.shape[3])

        if not self.difPanel.addKurtosis:
            fitter = self.dti_fitter
            ds     = self.dtiseries
            ds_res = self.dtiseries_res
        else:
            fitter = self.dki_fitter
            ds     = self.dkiseries
            ds_res = self.dkiseries_res
                
        ydata = fitter.dtifit(x, y, z)
        ds.setData(xdata, ydata)
        outSeries.append(ds)
        if self.difPanel.addResiduals:
            ydata = fitter.residuals(x, y, z)
            ds_res.setData(xdata, ydata)
            outSeries.append(ds_res)
            
        return outSeries



class DiffusionToolBar(plottoolbar.PlotToolBar):
    """The ``DiffusionToolBar`` is a toolbar for use with a
    :class:`.DiffusionPlotView`. It extends :class:`.PlotToolBar`
    mostly replicates :class:`.TimeSeriesToolBar`
    """

    @staticmethod
    def title():
        """Overrides :meth:`.ControlMixin.title`. Returns a title to be used
        in FSLeyes menus.
        """
        return 'Diffusion Plot toolbar'

    @staticmethod
    def supportedViews():
        """Overrides :meth:`.ControlMixin.supportedViews`. The
        ``DiffusionToolBar`` is only intended to be added to
        :class:`.DiffusionPlotView` views.
        """
        return [DiffusionPlotView]

    def __init__(self, parent, overlayList, displayCtx, tsPanel):
        """Create a ``DiffusionToolBar``.

        :arg parent:      The :mod:`wx` parent object.
        :arg overlayList: The :class:`.OverlayList` instance.
        :arg displayCtx:  The :class:`.DisplayContext` instance.
        :arg tsPanel:     The :class:`.TimeSeriesPanel` instance.
        """

        plottoolbar.PlotToolBar.__init__(
            self, parent, overlayList, displayCtx, tsPanel)

        self.togControl = actions.ToggleControlPanelAction(
            overlayList, displayCtx, tsPanel, DiffusionControlPanel)


        togControl = actions.ToggleActionButton(
            'togControl',
            actionKwargs={'floatPane': True},
            icon=[icons.findImageFile('spannerHighlight24'),
                  icons.findImageFile('spanner24')],
            tooltip='Show/hide the Diffusion control panel.')


        togList = actions.ToggleActionButton(
            'PlotListPanel',
            actionKwargs={'floatPane': True},
            icon=[icons.findImageFile('listHighlight24'),
                  icons.findImageFile('list24')],
            tooltip='Show/hide the plot list panel')

        togControl = props.buildGUI(self, self, togControl)
        togList = props.buildGUI(self, tsPanel, togList)

        self.InsertTools([togControl, togList], 0)

        nav = [togControl, togList] + self.getCommonNavOrder()

        self.setNavOrder(nav)



class DiffusionControlPanel(plotcontrol.PlotControlPanel):
    """Control panel for the Diffusion view.
    """

    @staticmethod
    def title():
        """Overrides :meth:`.ControlMixin.title`. Returns a title to be used
        in FSLeyes menus.
        """
        return 'Diffusion control panel'

    @staticmethod
    def supportedViews():
        """Overrides :meth:`.ControlMixin.supportedViews`.
        """
        return [DiffusionPlotView]

    def generateCustomPlotPanelWidgets(self, groupName):
        """Overrides :meth:`.PlotControlPanel.generateCustomPlotPanelWidgets`.

        Adds some widgets for controlling some properties of the
        :class:`.DiffusionPlotView`.
        """

        widgetList = self.getWidgetList()
        allWidgets = []
        difPanel    = self.plotPanel
        difProps    = ['plotMode',
                       'orderBy',
                       'showDTIFit',
                       'addKurtosis',
                       'addResiduals']

        for prop in difProps:
            widget = props.makeWidget(widgetList, difPanel, prop)
            allWidgets.append(widget)
            widgetList.AddWidget(
                widget,
                displayName=prop,
                groupName=groupName)

        return allWidgets


class DiffusionPlotView(plotpanel.OverlayPlotPanel):

    # This setting could be exposed to the user, allowing them to
    # choose how to plot the data ("X", "Y", and "Z" are arbitrary
    # placeholders).
    orderBy = props.Choice(('original','bval', 'V1'))
    """Options to reorder the data

    ============ ============================================
    ``original``  The data is not reordered
    ``bval``      The data is reordered according to the bval
    ``V1``        The data is reordered wrt abs(dot(V1,bvecs))
    ============ ============================================
    """
    plotMode = props.Choice(('original','normalise'))
    """Options to scale the plotted time courses.

    ================= =======================================================
    ``original``      The data is plotted with no modifications
    ``normalise``     The data is normalised by the lowest b-val.
    ================= =======================================================
    """
    showDTIFit   = props.Boolean(default=False)
    addKurtosis  = props.Boolean(default=False) 
    addResiduals = props.Boolean(default=False)
        
    
    @staticmethod
    def title():
        return "Diffusion plotting"

    @staticmethod
    def defaultLayout():
        """Returns a list of control panel types to be added for the default
        time series panel layout.
        """
        return ['DiffusionToolBar',
                'OverlayListPanel',
                'DiffusionInfoPanel',
                'PlotListPanel']


    @staticmethod
    def controlOrder():
        """Returns a list of control panel names, specifying the order in
        which they should appear in the  FSLeyes ortho panel settings menu.
        """
        return ['OverlayListPanel',
                'PlotListPanel',
                'DiffusionToolbar',
                'DiffusionInfoPanel',
                'DiffusionControlPanel']

    def __init__(self, parent, overlayList, displayCtx, frame):

        plotpanel.OverlayPlotPanel.__init__(self,
                                            parent,
                                            overlayList,
                                            displayCtx,
                                            frame,
                                            initialState=None)
        
        self.addListener('plotMode',   self.name, self.draw)
        self.addListener('orderBy',    self.name, self.draw)
        self.addListener('showDTIFit', self.name, self.draw)
        self.addListener('addKurtosis', self.name, self.draw)
        self.addListener('addResiduals', self.name, self.draw)

    def destroy(self):
        """Removes some listeners, and calls the :meth:`.PlotPanel.destroy`
        method.
        """

        self.removeListener('plotMode',   self.name)
        self.removeListener('orderBy',    self.name)
        self.removeListener('showDTIFit', self.name)
        self.removeListener('addKurtosis', self.name)
        self.removeListener('addResiduals', self.name)

        plotpanel.OverlayPlotPanel.destroy(self)

    def createDataSeries(self, overlay):

        displayCtx  = self.displayCtx
        overlayList = self.overlayList

        ovlDir = op.dirname(overlay.dataSource)

        # The below could be augmented to allow user to load the files
        #if not isinstance(overlay,Image):
        #    return None, None, None
        
        if not (op.exists(op.join(ovlDir, 'bvals')) and \
                op.exists(op.join(ovlDir, 'bvecs'))):
            return None, None, None
        else:
            bvals = np.loadtxt(op.join(ovlDir, 'bvals'))
            bvecs = np.loadtxt(op.join(ovlDir, 'bvecs'))
        if len(bvals) != overlay.shape[-1]:
            return None, None, None

        

        dsargs = [overlay, overlayList, displayCtx, self]

        ds        = DiffusionDataSeries(*dsargs)
        opts      = displayCtx.getOpts(overlay)
        targets   = [displayCtx, opts]
        propNames = ['location', 'volumeDim']

        ds.colour    = self.getOverlayPlotColour(overlay)
        ds.lineWidth = 2
        ds.lineStyle = '-'
        ds.alpha     = 1.0
        ds.label     = 'data'

        return ds, targets, propNames



    def prepareDataSeries(self,ds):
        """Overrides :class:`.PlotPanel.prepareDataSeries`. Given a
        :class:`.DataSeries` instance, scales and normalises the x and y data
        according to the current values of the :attr:`plotMode` and
        :attr:`orderBy` properties.        
        """
        opts         = self.displayCtx.getOpts(ds.overlay)
        x, y, z      = opts.getVoxel()
        xdata, ydata = ds.getData()

        if self.plotMode == 'normalise':
            idxs      = np.isclose(ds.bvals,0,atol=max(ds.bvals)/100.)
            orig_data = ds.fitter.image[x,y,z,:]
            ydata     = ydata / orig_data[idxs].mean()

        idxs = np.arange( len(ds.bvals) )

        if self.orderBy == 'bval':
            # sort data by bvals, low to high
            idxs  = np.argsort(ds.bvals)

        elif self.orderBy == 'V1':
            v1   = ds.fitter.v1(x, y, z)
            # below hack to order by both bvals and v1
            dp   = np.abs(np.dot(v1,ds.bvecs.T))      
            idxs = np.argsort(bvals_round(ds.bvals)*1e3+dp)

        ydata = ydata[idxs]

        return xdata, ydata




    def draw(self, *a):
        """Overrides :meth:`.OverlayPlotPanel.draw`. Draws some
        :class:`.DiffusionDataSeries` using the
        :meth:`.PlotPanel.drawDataSeries` method.
        """

        if not self or self.destroyed:
            return

        canvas = self.canvas
        pss    = self.getDataSeriesToPlot()

        canvas.drawDataSeries(extraSeries=pss)
        canvas.drawArtists()




class DiffusionInfoPanel(ctrlpanel.ControlPanel):

    @staticmethod
    def supportedViews():
        return [DiffusionPlotView]

    @staticmethod
    def defaultLayout():
        """Returns a dictionary of arguments to be passed to the
        :meth:`.ViewPanel.togglePanel` method.
        """
        return {'location' : wx.LEFT}

    def __init__(self, parent, overlayList, displayCtx, viewPanel):
        ctrlpanel.ControlPanel.__init__(
            self, parent, overlayList, displayCtx, viewPanel)

        self.__info = wxhtml.HtmlWindow(self)

        self.__sizer = wx.BoxSizer(wx.VERTICAL)
        self.__sizer.Add(self.__info, flag=wx.EXPAND, proportion=1)
        self.SetSizer(self.__sizer)

        displayCtx.addListener('location', self.name, self.__locationChanged)
        viewPanel.addListener('addKurtosis', self.name, self.__locationChanged)

        self.SetMinSize((200, 200))
        self.Layout()
        self.__locationChanged()


    def destroy(self):
        self.displayCtx.removeListener('location', self.name)
        super().destroy()


    def __locationChanged(self, *a):
        info = ''
        for ovl in self.overlayList:

            ds = self.viewPanel.getDataSeries(ovl)
            if ds is None:
                continue

            opts    = self.displayCtx.getOpts(ovl)
            x, y, z = opts.getVoxel()

            if not self.viewPanel.addKurtosis:
                fitter = ds.dti_fitter
                k      = None
            else:
                fitter = ds.dki_fitter
                k      = fitter.kurt(x, y, z)
                
            md = fitter.md(x, y, z)
            fa = fitter.fa(x, y, z)
            v1 = fitter.v1(x, y, z)
            

            info += f'<b>{ovl.name} [{x} {y} {z}]</b>'
            info += '<ul>'
            info += f'<li>FA   : {fa:0.2f}</li>'
            info += f'<li>MD   : {md:0.2f} (um^2/ms)</li>'
            info += f'<li>V1   : [{v1[0]:0.2f},{v1[1]:0.2f},{v1[2]:0.2f}]</li>'
            if k is not None:
                info += f'<li>Kurt   : {k:0.2f}</li>'
            info += '</ul>'
            info += '<hr><p><b>Shells</b><br>'
            for shell,dirs in ds.shells.items():
                info += f'b = {int(shell)} : {int(dirs)} dirs<br>'

        self.__info.SetPage(info)
