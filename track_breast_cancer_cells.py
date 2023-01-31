from types import SimpleNamespace
import pickle
import ipdb
import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG)

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage.io import imread,imsave
from scipy.ndimage import label
import networkx as nx
from scipy.optimize import linear_sum_assignment
import napari

from segtools import nhl_tools, track_tools, trackmeasure


def load():
  img  = imread("raw/Breast cancer cell dataset/Test data/R1_cell migration R1 - Position 79_XY1562686175_Z0_T00_C1.tif")
  lab  = imread("raw/R1_cell migration R1 - Position 79_XY1562686175_Z0_T00_C1.stardist.tif")
  nhls = nhl_tools.labs2nhls(lab, img)
  
  R = SimpleNamespace()
  R.img = img
  R.lab = lab
  R.nhls = nhls

  for i in range(1,10):
    R.__dict__[f'res{i:02d}'] = pickle.load(open(f"ch3_track_breastCancerCells/track_breastCancerCells-run-ss{i:02d}.pkl", 'rb'))

  return R

def run_shear(S, subsample=None):

  img  = imread("raw/Breast cancer cell dataset/Test data/R1_cell migration R1 - Position 79_XY1562686175_Z0_T00_C1.tif")
  lab  = imread("raw/R1_cell migration R1 - Position 79_XY1562686175_Z0_T00_C1.stardist.tif")
  nhls = nhl_tools.labs2nhls(lab, img)

  if subsample is not None:
    img  = img[::subsample]
    lab  = lab[::subsample]
    nhls = nhls[::subsample]

  def vertcost(vert):
      # {'label': 150, 'area': 1, 'centroid': (291.0, 765.0), 'bbox': (291, 765, 292, 766), 'dims': [1, 1], 'slice': (slice(291, 292, None), slice(765, 766, None))}
      return -1

  def edgecost(vert1,vert2):
      return -1

  """
  Params
  trtest  : ec=0,vc=0,knn_n=15,knn_dub=50,edgecost=None,vertcost=None,allow_div=False,allow_exit=False,allow_appear=False,do_velcorr=True,neib_edge_cutoff=40,velgrad_scale=20,on_edges=None {f1_det:0.978 , f1_tra:0.964}
  trtest2 : ec=-1,vc=-1,knn_n=2,knn_dub=50,edgecost=None,vertcost=None,allow_div=True,allow_exit=False,allow_appear=False,do_velcorr=True,neib_edge_cutoff=50,velgrad_scale=20,on_edges=None {f1_det:1.000 , f1_tra:0.996}
  """
  R = SimpleNamespace()
  R.shear = SimpleNamespace()
  R.shearalign = SimpleNamespace()
  R.nn = SimpleNamespace()
  R.nnalign = SimpleNamespace()
  R.tba = SimpleNamespace()
  R.tbaalign = SimpleNamespace()

  if Shear:
    factory = track_tools.TrackFactory(vertcost=vertcost,edgecost=edgecost,)
    alltracks = [factory.nhls2tracking(nhls[i:i+2]) for i in range(len(nhls)-1)]
    tr = track_tools.compose_trackings(factory, alltracks, nhls)
    pickle.dump(tr, open("data-mid/trtest2-noshear.pkl",'wb'))
    tr = pickle.load(open('data-mid/trtest2-noshear.pkl','rb'))
    # track_tools.stats_tr(tr)
    ltpmap = [{v['label'] : v['centroid'] for v in nhl} for nhl in nhls]
    for n in tr.tb.nodes:
      tr.tb.nodes[n]['time'] = n[0]
      tr.tb.nodes[n]['pt']   = ltpmap[n[0]][n[1]]
    track_tools._tb_add_track_labels(tr.tb)
    nap = track_tools.tb2nap(tr.tb)
    R.shear.tb = tr.tb
    R.shear.nap = nap

  if ShearAlign:
    factory = track_tools.TrackFactory(vertcost=vertcost,edgecost=edgecost, )
    labels = [np.array([n['label'] for n in nhl]) for nhl in nhls]
    pts = [np.array([n['centroid'] for n in nhl]) for nhl in nhls]
    pts_mean = [x.mean(0) for x in pts]
    pts_mean = np.mean(pts_mean,0)
    pts_centered = [p - p.mean(0) + pts_mean for p in pts]
    for i,nhl in enumerate(nhls):
      for j,n in enumerate(nhl):
        n['centroid'] = pts_centered[i][j]
        assert labels[i][j]==n['label']

    # ipdb.set_trace()

    alltracks = [factory.nhls2tracking(nhls[i:i+2]) for i in range(len(nhls)-1)]
    tr = track_tools.compose_trackings(factory, alltracks, nhls)

    ## undo the center-of-mass alignment for NHLS and TB
    for i,nhl in enumerate(nhls):
      for j,n in enumerate(nhl):
        n = nhls[i][j]
        n['centroid'] = pts[i][j]

    for (t,l) in tr.tb.nodes:
      j = (labels[t]==l).argmax()
      # l = n['label']
      tr.tb.nodes[(t,l)]['pt'] = pts[t][j]

    pickle.dump(tr, open("data-mid/trtest2-shear.pkl",'wb'))
    tr = pickle.load(open('data-mid/trtest2-shear.pkl','rb'))
    # track_tools.stats_tr(tr)
    ltpmap = [{v['label'] : v['centroid'] for v in nhl} for nhl in nhls]
    for n in tr.tb.nodes:
      tr.tb.nodes[n]['time'] = n[0]
      tr.tb.nodes[n]['pt']   = ltpmap[n[0]][n[1]]
    track_tools._tb_add_track_labels(tr.tb)
    nap = track_tools.tb2nap(tr.tb)
    R.shearalign.tb = tr.tb
    R.shearalign.nap = nap

  R.img = img
  # T = SimpleNamespace(**locals())

  if subsample is None: subsample=0
  pickle.dump(R,open(f"data-mid/track_breastCancerCells-run-ss{subsample:02d}.pkl" , 'wb'))

  return R

def run_nn(S, subsample=None):

  ## local `img` doesn't overwrite S.img, does it?
  # img  = S.img
  # lab  = S.lab
  # nhls = S.nhls

  if subsample is None: subsample = 1

  img  = S.img[::subsample]
  lab  = S.lab[::subsample]
  nhls = S.nhls[::subsample]

  ## run in normal mode
  ltps2 = [np.array([n['centroid'] for n in nhl]) for nhl in nhls]
  tb2   = track_tools.nn_tracking_on_ltps(ltps2,scale=[1,1])
  nap2  = track_tools.tb2nap(tb2)

  ## run in C.O.M. aligned mode
  means = [np.mean(arr,0) for arr in ltps2]
  meanymeany = np.mean(means,axis=0)
  ltps3 = [ltps2[i]-means[i]+meanymeany for i in range(len(means))]
  tb3   = track_tools.nn_tracking_on_ltps(ltps3,scale=[1,1])
  for n in tb3.nodes:
    p = tb3.nodes[n]['pt']
    p = p + means[n[0]] - meanymeany
    tb3.nodes[n]['pt'] = p
  nap3  = track_tools.tb2nap(tb3)


  R = SimpleNamespace()
  R.nn = SimpleNamespace()
  R.nnalign = SimpleNamespace()
  R.nn.tb = tb2
  R.nn.nap = nap2
  R.nnalign.tb  = tb3
  R.nnalign.nap = nap3

  return R

def view(T,viewer):
  # viewer.add_image(T.img)
  # viewer.add_tracks(T.shearalign.nap.tracklets,  properties=T.shearalign.nap.properties,  graph=T.shearalign.nap.graph,  scale=[1,1], name='shear costs')
  # viewer.add_tracks(T.shear.nap.tracklets,  properties=T.shear.nap.properties,  graph=T.shear.nap.graph,  scale=[1,1], name='shear costs')
  viewer.add_tracks(T.nn.nap.tracklets, properties=T.nn.nap.properties, graph=T.nn.nap.graph, scale=[1,1], name='nearest')
  viewer.add_tracks(T.nnalign.nap.tracklets, properties=T.nnalign.nap.properties, graph=T.nnalign.nap.graph, scale=[1,1], name='nearest-align')

## Initial results saved in `linkscores.txt`.
## Second round saved in `ch3-breastcancer-linkscores2.txt`
def scores():
  # T = run(subsample=None)
  # T = pickle.load(open("t1.pkl",'rb'))
  # print("\n== nnalign scores ==")
  # trackmeasure.compare_tb(T.nnalign.tb,T.nnalign.tb)
  # print("\n== nn scores ==")
  # trackmeasure.compare_tb(T.nnalign.tb,T.nn.tb)
  # print("\n== shear scores ==")
  # trackmeasure.compare_tb(T.nnalign.tb,T.shear.tb)
  # print("\n== shear align scores ==")
  # trackmeasure.compare_tb(T.nnalign.tb,T.shearalign.tb)

  allscores = []

  ## we don't actually have real GT ! so we use this as a proxy
  gt = pickle.load(open(f"data-mid/mpl/ch3_track_breastCancerCells/track_breastCancerCells-run-ss01.pkl",'rb')).nnalign.tb

  for ss in range(1,10):
    # T = run(subsample=None)
    # T = R.__dict__[f'res{ss:02d}']
    T = pickle.load(open(f"data-mid/mpl/ch3_track_breastCancerCells/track_breastCancerCells-run-ss{ss:02d}.pkl",'rb'))

    if ss>1:
      gt_ss = trackmeasure.subsample_graph(gt,subsample=ss)
    else:
      gt_ss = gt

    allscores.append( trackmeasure.compare_tb(gt_ss , T.nnalign.tb   )[3].f1_tra )
    allscores.append( trackmeasure.compare_tb(gt_ss , T.nn.tb        )[3].f1_tra )
    allscores.append( trackmeasure.compare_tb(gt_ss , T.shear.tb     )[3].f1_tra )
    allscores.append( trackmeasure.compare_tb(gt_ss , T.shearalign.tb)[3].f1_tra )

  return allscores



## See `linkscores1.txt`
## Scores cycle between these four,
## 1. nnalign scores
## 2. nn scores
## 3. shear scores
## 4. shear align scores
f1tra_linkscores_1 = [1.00000,
  0.99751,
  0.99537,
  0.99550,
  0.97934,
  0.97025,
  0.98892,
  0.98808,
  0.92628,
  0.85993,
  0.97169,
  0.97514,
  0.86317,
  0.70446,
  0.90869,
  0.95705,
  0.75822,
  0.55276,
  0.76455,
  0.90970,
  0.68961,
  0.42753,
  0.60490,
  0.85307,
  0.59530,
  0.33565,
  0.44955,
  0.78199,
  0.51066,
  0.26618,
  0.31947,
  0.66577,
  0.44821,
  0.22168,
  0.2412,
  0.56561,
  ]

## Scores cycle between these four,
## 1. nnalign scores
## 2. nn scores
## 3. shear scores
## 4. shear align scores
f1tra_linkscores_2 = [1.0,
 0.9975131975044719,
 0.9815686447015353,
 0.9817006114459156,
 0.9793395726646654,
 0.9702454529401378,
 0.9728494623655914,
 0.9719994623896779,
 0.9262769878883623,
 0.8599262769878884,
 0.9613653161325002,
 0.9607632356893309,
 0.863167104111986,
 0.7044619422572178,
 0.9389790552180615,
 0.9414210811788103,
 0.7582229816317813,
 0.552755232806493,
 0.9177257525083612,
 0.9293196603815195,
 0.6896103896103896,
 0.42753246753246754,
 0.9127679654846973,
 0.9158349069328298,
 0.5952955367913149,
 0.3356453558504222,
 0.8815270547404953,
 0.8782984167599552,
 0.5106617647058823,
 0.2661764705882353,
 0.7712062256809339,
 0.8197964478148074,
 0.4482063683998388,
 0.22168480451430875,
 0.6357298474945534,
 0.7990492653414002]


def plots():

  plt.figure()
  # plt.plot(f1tra_linkscores_1[0::4], label="nnalign - 1")
  # plt.plot(f1tra_linkscores_1[1::4], label="nn - 1")
  # plt.plot(f1tra_linkscores_1[2::4], label="shear - 1")
  # plt.plot(f1tra_linkscores_1[3::4], label="shear - 1")

  plt.plot(f1tra_linkscores_2[0::4], label="nn aligned")
  plt.plot(f1tra_linkscores_2[1::4], label="nn")
  plt.plot(f1tra_linkscores_2[2::4], label="strain")
  plt.plot(f1tra_linkscores_2[3::4], label="strain aligned")

  plt.legend()



