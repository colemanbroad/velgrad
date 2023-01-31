import pickle
import numpy as np
from types import SimpleNamespace
from skimage import io
from scipy.ndimage import label

from segtools import nhl_tools, track_tools, trackmeasure
import networkx as nx
import matplotlib.pyplot as plt

import ipdb
from matplotlib.colors import ListedColormap

from glob import glob
import os
import ctypes

from numpy.ctypeslib import ndpointer


def loadlib():
  # base = "/Users/broaddus/Desktop/phd/thesis/src/zig/"
  base = "/Users/broaddus/Desktop/work/zig-tracker/"
  files = glob(base + "zig-cache/o/*/libtrack.dylib")
  files.sort(key=lambda x: os.path.getmtime(x))
  return ctypes.CDLL(files[-1])

lib = loadlib()

def strain_track(va,vb):
  va = va.astype(np.float32)
  vb = vb.astype(np.float32)
  va_ = va.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  vb_ = vb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  parents = np.zeros(len(vb), dtype=np.int32)
  res_ = parents.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  # ipdb.set_trace()
  err = lib.strain_track2d(va_ , len(va) , vb_ , len(vb) , res_)
  if (err!=0): print("ERRORRRRRROROR")
  return parents

def greedy_track(va,vb):
  va = va.astype(np.float32)
  vb = vb.astype(np.float32)
  va_ = va.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  vb_ = vb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
  parents = np.zeros(len(vb), dtype=np.int32)
  res_ = parents.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  # err = lib.greedy_track2d.restype = ndpointer(dtype=ctypes.c_int32, shape=(len(vb),))
  err = lib.greedy_track2d(va_,len(va),vb_,len(vb), res_)
  if (err!=0): print("ERRORRRRRROROR")
  return parents


def build_S():

  img     = io.imread('data-raw/img003.tif')
  imgmean = img.mean((1,2), keepdims=True)
  img     = img/imgmean
  pimg    = np.load('data-raw/img003_Probabilities.npy')[...,0]
  lab     = np.array([label(x > 0.5)[0] for x in pimg])
  nhls    = nhl_tools.labs2nhls(lab, img)

  S = SimpleNamespace()
  S.img = img
  S.lab = lab
  S.pimg = pimg
  S.nhls = nhls  
  S.ltps = [np.array([n['centroid'] for n in nhl]) for nhl in nhls]
  return S

## Build a GT true-branching from our special annotation labels
def add_gt_from_viewer(S,viewer=None):
  if viewer:
    gt = viewer.layers["img003-labels"].data
  else:
    gt = np.load("data-anno/img003-gt.npy")
  print(np.unique(gt))

  S.gt  = gt

  ## first build graphs
  G = SimpleNamespace()
  G.pts = dict()
  G.edges = dict()
  for t,lab_t in enumerate(S.gt):
    nhl = nhl_tools.hyp2nhl(lab_t,)
    for n in nhl:
      l = n['label']
      if t!=0: 
        G.edges[(t,l)] = (t-1,l)
      G.pts[(t,l)] = n['centroid']

  G.tb = nx.from_edgelist(list(G.edges.items()), create_using=nx.DiGraph)
  G.tb = G.tb.reverse()

  for n in G.tb.nodes:
    G.tb.nodes[n]["pt"] = G.pts[n]
    print(G.tb.nodes[n])

  track_tools._tb_add_track_labels(G.tb)
  S.gt_G = G

## Compute the proposed tracking by using TbA on the ilastik detections
def add_tba_tracking(S):
  nhls = S.nhls
  def vertcost(vert): return -1
  def edgecost(vert1,vert2): 
    # return -1
    u0 = np.array(vert1['centroid'])
    u1 = np.array(vert2['centroid'])
    return np.linalg.norm((u1-u0)/30)**2 - 1.0
    # return 0
  def straincost(u0, u1, v0, v1):
    du  = np.array(u1) - np.array(u0)
    dv  = np.array(v1) - np.array(v0)
    res = (np.linalg.norm(du - dv)/20) - 1.0
    return res

  factory = track_tools.TrackFactory(vertcost=vertcost, edgecost=edgecost, straincost=None, knn_n=10, allow_exit=True, allow_appear=True, neib_edge_cutoff=70, allow_div=True)
  # alltracks = [factory.nhls2tracking(nhls[i:i+2]) for i in range(len(nhls)-1)]
  tr = factory.nhls2tracking(nhls)
  # tr = track_tools.compose_trackings(factory, alltracks, nhls)
  # alltracksrev = [track_tools.nhls2tracking(nhls[i:i+2][::-1]) for i in range(len(nhls)-2)]
  track_tools.stats_tr(tr)

  ltpmap = [{v['label'] : v['centroid'] for v in nhl} for nhl in S.nhls]
  for n in tr.tb.nodes:
    tr.tb.nodes[n]['time'] = n[0]
    tr.tb.nodes[n]['pt']   = ltpmap[n[0]][n[1]]
  track_tools._tb_add_track_labels(tr.tb)

  S.test = tr
  """
  || name             || knn || vert || edge        || strain           || extra
  || test             || 10  || -1   || |d/30|^2 -1 || None             || in-place updates
  || tr_10            || 10  || -1   || |d/20|^2 -1 || None             || (identical to tr_4)
  || * tr_9           || 10  || -1   || -1          || |dx/20| - 1      || neib_edge_cutoff=70 (default)
  || tr_8             || 10  || -1   || -1          || |dx/40| - 1      || neib_edge_cutoff=70
  || tr_7             || 10  || -1   || -1          || |dx/70| - 1      || neib_edge_cutoff=70
  || tr_6             || 20  || -1   || |d/50|^2 -1 || None             || 
  || tr_5             || 10  || -1   || |d/20|^2 -1 || |d/40| -1        || 
  || tr_4             || 10  || -1   || |d/20|^2 -1 || None             || 
  || tr_3             || 10  || -1   || |d/80|^2 -1 || None             || 
  || tr_2             || 10  || -1   || |d/40|^2 -1 || None             || 
  || tr_1             || 10  ||  0   || 0           || (dx/40)^2 - 1    || 
  || tr_knn10_e2      || 10  || -1   || -1          || (dx/40)^2 - 1    || 
  || tr_knn10_novg    || 10  || -1   || -1          || None             || 
  || tr_knn10         || 10  || -1   || -1          || |dx/40| - 1      || 
  || tr_knn4          || 4   || -1   || -1          || |dx/40| - 1      || 
  || tr_knn6          || 6   || -1   || -1          || |dx/40| - 1      || exit=False enter=False
  || tr               || 2   || -1   || -1          || |dx/40| - 1      || exit=False enter=False

  !!! orig strain cost may have been |dx/20| - 1 with neib_edge_cutoff=40
  !!! [^1]: i'm not sure about these. may have been |dx/20| which was the original
  """

## Compute Nearest Neighbour tracking with distance-upper-bound
def add_nn_tracking(S):
  S.nn_tb = track_tools.nn_tracking_on_ltps(S.ltps,scale=[1,1],dub=50)

import napari

def viewer_add_tb(viewer,tb, name='kt_1'):
  nap = track_tools.tb2nap(tb)
  nap.properties['const'] = np.ones(nap.tracklets.shape[0])
  # nap.properties['const'][:-1] = 1
  # nap.properties['time'] = nap.tracklets[:,1]

  # ipdb.set_trace()
  # cmap = np.zeros([100,3])
  # cmap[:] = (255,255,0) ## yellow
  # cmap = ListedColormap(cmap)
  # cmap = napari.utils.colormaps.Colormap(cmap)
  # cmap = 'red'
  cmap = 'gray_r'
  # cmap = napari.utils.colormaps
  # cmap = napari.utils.colormaps.label_colormap()
  # cmap = napari.utils.colormaps.Colormap([(1, 1, 1, 1)]*(nap.tracklets.shape[0]+1), interpolation="zero")
  # cmap_dict = {'time':cmap}
  # cmap = 'blue'
  viewer.add_tracks(nap.tracklets, 
                    graph=nap.graph, 
                    properties=nap.properties, 
                    name=name, 
                    colormap=cmap, 
                    blending='opaque', 
                    tail_length=1, 
                    tail_width=5,
                    # colormaps_dict=cmap_dict,
                    )

  # e0 = np.array([tb.nodes[e[0]]["pt"] for e in tb.edges])
  # e1 = np.array([tb.nodes[e[1]]["pt"] for e in tb.edges])
  # vecs = np.stack([e0,e1],axis=1)
  # ipdb.set_trace()
  # viewer.add_vectors(vecs)

def add_greedy_boring(S):
  parents_list = []
  for i in range(len(S.ltps)-1):
    va = S.ltps[i]
    vb = S.ltps[i+1]
    
    const = va.mean()
    va = va/const
    vb = vb/const

    parents = greedy_track(va,vb)
    parents_list.append(parents)

  ## fast, boring
  S.fb_tb = track_tools._parents2tb(parents_list,S.ltps)

def add_greedy_strain(S):
  parents_list = []
  for i in range(len(S.ltps)-1):
    va = S.ltps[i]
    vb = S.ltps[i+1]
    
    const = va.mean()
    va = va/const
    vb = vb/const

    parents = strain_track(va,vb)
    parents_list.append(parents)

  ## greedy, strain
  S.gs_tb = track_tools._parents2tb(parents_list,S.ltps)


def plot_compare_link_scores_over_time(S,viewer=None):
  gt_splits = []
  for i in range(8-1): ## t0
    g0 = S.gt_G.tb.subgraph([n for n in S.gt_G.tb.nodes if n[0] in {i,i+1}])
    gt_splits.append(g0)

  tbs = [
         (S.gs_tb,              "GreedyStrain",) ,
         (S.fb_tb,              "Fast Boring",) ,
         (S.nn_tb,              "NN",) ,
         # (S.tr_9.tb,            "TbA + Strain",) ,
         # (S.tr.tb,            "strain-2",) ,
         # (S.tr_knn6.tb,       "strain-6",) ,
         # (S.tr_knn4.tb,       "strain-4",) ,
         # (S.tr_knn10.tb,      "pure strain orig",) ,
         # (S.tr_knn10_novg.tb, "10 - no vg",) ,
         # (S.tr_knn10_e2.tb,   "10 - Quad",) ,
         # (S.tr_1.tb,          "0 V/E cost",) ,
         # (S.tr_4.tb,            "dist costs 20",) ,
         # (S.tr_3.tb,            "dist costs 80",) ,
         # (S.tr_2.tb,            "dist costs 40",) ,
         # (S.tr_6.tb,            "dist costs 50",) ,
         # (S.tr_5.tb,          "strain+dist costs",) ,
         # (S.tr_7.tb,          "pure strain 70",) ,
         # (S.tr_8.tb,          "pure strain 40",) ,
         # (S.tr_10.tb,           "dist costs 20 (2)"),
         (S.test.tb,            "TbA"),
         ]
  
  plt.figure()
  for k,(tb,label) in enumerate(tbs):
    scores = []
    for i in range(8-1):
      g = tb.subgraph([n for n in tb.nodes if n[0] in {i,i+1}])
      scores.append(trackmeasure.compare_tb(gt_splits[i], g)[3].recall_tra)
      # if label=="dist costs 20":
      #   plt.plot([s+0.01 for s in scores], '-o', alpha=1-0.05*k, label=label)
      # else:
    scores = [s + np.random.rand()*0.01 - 0.005 for s in scores]
    plt.plot(scores, '-o', alpha=1-0.05*k, label=label)
    if viewer:
      viewer_add_tb(viewer, tb, label)

  plt.legend()

def rerun():
  S = build_S()
  add_gt_from_viewer(S)
  # add_nn_tracking(S)
  # add_tba_tracking(S)
  add_greedy_strain(S)
  # add_greedy_boring(S)
  return S

if __name__=="__main__":
  S = rerun()