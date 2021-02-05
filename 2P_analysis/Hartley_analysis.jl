# Attempting to load a primate dataset: Callaway lab, Salk Institute
# Author - Abhishek De, copied from 2PResponseRF.jl

# Loading all the necessary Julia libraries
using NeuroAnalysis,Statistics,StatsBase,StatsPlots,DataFrames,DataFramesMeta,Mmap,Images,Interact
using CSV,MAT,DataStructures,HypothesisTests,StatsFuns,Random,Plots, KernelDensity, Gadfly, LinearAlgebra

##--- Some important parameters
interpolatedData = true   # If you have multiplanes. True: use interpolated data; false: use uniterpolated data. Results are slightly different.
hartelyBlkId = 5641
stanorm = nothing
stawhiten = nothing
delays = -0.066:0.033:0.4
print(collect(delays))
isplot = false

folder = "F:NHP_data"
subject = "AF4"
subfolder = "2P_data_processed"
recordSession = "004" # Unit
testId = "001"  # Stimulus test

# Location of the .segment, .signals and the .mat (metadata) files for Abhishek

segmentfile_id = "F:NHP_data\\AF4\\2P_data_processed\\U004\\004_001\\AF4_004_001_merged.segment"
signalfile_id = "F:NHP_data\\AF4\\2P_data_processed\\U004\\004_001\\AF4_004_001_merged.signals"
metafile_id  = "F:NHP_data\\AF4\\2P_data_processed\\U004\\AF4_004_001_ot_meta.mat"

# Loading the meta file and storing it in "dataset"
metadataset = prepare(metafile_id)
ex = metadataset["ex"]
envparam = ex["EnvParam"]
coneType = getparam(envparam,"colorspace")
sbx = metadataset["sbx"]["info"]
sbxft = ex["frameTimeSer"]   # time series of sbx frame in whole recording

# Condition Tests
envparam = ex["EnvParam"];
preicidur = ex["PreICI"];
conddur = ex["CondDur"];
suficidur = ex["SufICI"]
condon = ex["CondTest"]["CondOn"]
condoff = ex["CondTest"]["CondOff"]
condidx = ex["CondTest"]["CondIndex"]
# condtable = DataFrame(ex["Cond"])
condtable =  DataFrame(ex["raw"]["log"]["randlog_T1"]["domains"]["Cond"])
rename!(condtable, [:oridom, :kx, :ky,:bwdom,:colordom])

# find out blanks and unique conditions
blkidx = condidx .>= hartelyBlkId  # blanks start from 5641
cidx = .!blkidx
condidx2 = condidx.*cidx + blkidx.* hartelyBlkId

# Loading the .segment and the .signal file
segment = prepare(segmentfile_id)
signal = prepare(signalfile_id)
sig = transpose(signal["sig"])   # 1st dimention is cell roi, 2nd is fluorescence trace
spks = transpose(signal["spks"])  # 1st dimention is cell roi, 2nd is spike train

# Setting up the params: taken from 2PbatchTests.jl
param = Dict{Any,Any}(
    :dataexportroot => "F:\\AF2\\2P_analysis\\Summary\\DataExport",
    :interpolatedData => true,   # If you have multiplanes. True: use interpolated data; false: use uniterpolated data. Results are slightly different.
    :preOffset => 0.1,
    :responseOffset => 0.05,  # in sec
    :α => 0.05,   # p value
    :sampnum => 100,   # random sampling 100 times
    :fitThres => 0.5,
    :hueSpace => "HSL",   # Color space used? DKL or HSL
    :diraucThres => 0.8,   # if passed, calculate hue direction, otherwise calculate hue axis
    :oriaucThres => 0.5,
    :Respthres => 0.1,  # Set a response threshold to filter out low response cells?
    :blankId => 36,  # Blank Id; AF3AF4=36, AE6AE7=34
    :excId => [27,28])  # Exclude some condition?

param[:model]=[:STA]
param[:stanorm] = nothing
param[:stawhiten] = nothing
param[:downsample] = 1  # for down-sampling stimuli image, 1 is no down-sampling
param[:hartleynorm] = false
param[:hartleyscale] = 1
param[:hartelyBlkId]=5641
param[:delayLB] = -0.066  # in sec; Usually do not need to change it
param[:delayUB] = 0.4   # in sec; Usually do not need to change it
param[:delayStep] = 0.033  # Bidirectional = 0.033 sec, Unidirectional = 0.066 sec
param[:delays] = param[:delayLB]:param[:delayStep]:param[:delayUB]
param[:maskradius] = 1 #AE6=0.75 AE7=0.24 AF2=1 AF3=0.6 AF4=0.65

# Prepare Imageset
downsample = haskey(param,:downsample) ? param[:downsample] : 2
sigma = haskey(param,:sigma) ? param[:sigma] : 1.5
bgcolor = RGBA([0.5,0.5,0.5,1]...)
coneType = string(getparam(envparam,"colorspace"))
masktype = getparam(envparam,"mask_type")
maskradius = getparam(envparam,"mask_radius")
masksigma = 1#getparam(envparam,"Sigma")
hartleyscale = haskey(param,:hartleyscale) ? param[:hartleyscale] : 1
hartleynorm = haskey(param, :hartleynorm) ? param[:hartleynorm] : false
xsize = getparam(envparam,"x_size")
ysize = getparam(envparam,"y_size")
stisize = xsize
ppd = haskey(param,:ppd) ? param[:ppd] : 52
ppd = ppd/downsample
imagesetname = "Hartley_stisize$(stisize)_hartleyscalescale$(hartleyscale)_ppd$(ppd)"
maskradius = maskradius /stisize + 0.03

if !haskey(param,imagesetname)
    imageset = Dict{Any,Any}(:image =>map(i->GrayA.(hartley(kx=i.kx,ky=i.ky,bw=i.bwdom,stisize=stisize, ppd=ppd,norm=hartleynorm,scale=hartleyscale)),eachrow(condtable)))
    # imageset = Dict{Any,Any}(:image =>map(i->GrayA.(grating(θ=deg2rad(i.Ori),sf=i.SpatialFreq,phase=rem(i.SpatialPhase+1,1)+0.02,stisize=stisize,ppd=23)),eachrow(condtable)))
    # imageset = Dict{Symbol,Any}(:pyramid => map(i->gaussian_pyramid(i, nscale-1, downsample, sigma),imageset))
    imageset[:sizepx] = size(imageset[:image][1])
    param[imagesetname] = imageset
end

# Prepare Image Stimuli
imageset = param[imagesetname]
bgcolor = oftype(imageset[:image][1][1],bgcolor)
imagestimuliname = "bgcolor$(bgcolor)_masktype$(masktype)_maskradius$(maskradius)_masksigma$(masksigma)" # bgcolor and mask define a unique masking on an image set
if !haskey(imageset,imagestimuliname)
    imagestimuli = Dict{Any,Any}(:stimuli => map(i->alphablend.(alphamask(i,radius=maskradius,sigma=masksigma,masktype=masktype)[1],[bgcolor]),imageset[:image]))
    imagestimuli[:unmaskindex] = alphamask(imageset[:image][1],radius=maskradius,sigma=masksigma,masktype=masktype)[2]
    imageset[imagestimuliname] = imagestimuli
end
imagestimuli = imageset[imagestimuliname]

## Load data
planeNum = size(segment["mask"],3)  # how many planes
if interpolatedData
    planeStart = vcat(1, length.(segment["seg_ot"]["vert"]).+1)
end

# Trying to calculate the plain vanilla STA
pn = 1; # Trying for just the first plane

if interpolatedData
    cellRoi = segment["seg_ot"]["vert"][pn]
else
    cellRoi = segment["vert"]
end
cellNum = length(cellRoi)
display("Cell Number: $cellNum")

if interpolatedData
    # rawF = sig[planeStart[pn]:planeStart[pn]+cellNum-1,:]
    spike = spks[planeStart[pn]:planeStart[pn]+cellNum-1,:]
else
    # rawF = sig
    spike = spks
end

imagesize = imageset[:sizepx]
xi = imagestimuli[:unmaskindex]

##--- estimating RF using STA: Meat of this code

uci = unique(condidx2)
ucii = map(i->findall(condidx2.==i),deleteat!(uci,findall(isequal(hartelyBlkId),uci)))   # find the repeats of each unique condition
ubii = map(i->findall(condidx2.==i), [hartelyBlkId]) # find the repeats of each blank condition

uy = Array{Float64}(undef,cellNum,length(delays),length(ucii))
ucy = Array{Float64}(undef,cellNum,length(delays),length(ucii))
uby = Array{Float64}(undef,cellNum,length(delays),length(ubii))

usta = Array{Float64}(undef,cellNum,length(delays),length(xi))
cx = Array{Float64}(undef,length(ucii),length(xi))

foreach(i->cx[i,:]=gray.(imagestimuli[:stimuli][uci[i]][xi]),1:size(cx,1))
for d in eachindex(delays)

    display("Processing delay: $d")
    y,num,wind,idx = epochspiketrain(sbxft,condon.+delays[d], condoff.+delays[d],isminzero=false,ismaxzero=false,shift=0,israte=false)
    spk=zeros(size(spike,1),length(idx))

    for i =1:length(idx)
        spkepo = @view spike[:,idx[i][1]:idx[i][end]]
        spk[:,i]= mean(spkepo, dims=2)
    end

    for cell in 1:cellNum
        # display(cell)
        cy = map(i->mean(spk[cell,:][i]),ucii)  # response to grating
        bly = map(i->mean(spk[cell,:][i]),ubii) # response to blank, baseline
        ry = cy.-bly  # remove baseline
        csta = sta(cx,ry,norm=stanorm,whiten=stawhiten)  # calculate sta
        ucy[cell,d,:]=cy
        uby[cell,d,:]=bly
        uy[cell,d,:]=ry
        usta[cell,d,:]=csta

        if isplot
            r = [extrema(csta)...]
            title = "Unit_$(cell)_STA_$(delays[d])"
            p = plotsta(csta,sizepx=imagesize,sizedeg=stisize,ppd=ppd,index=xi,title=title,r=r)
        end
    end
end

## SNR analysis on STA
numcells = cellNum

# Calculating the energy
energy = reshape([mean(usta[cell,d,:].^2) for cell in 1:numcells for (d,_) in enumerate(delays)], numcells, :)

# Estimating the noise (standard deviation) from the noise frame
noise = [std(usta[cell,3,:]) for cell in 1:numcells]
maxdelay = [argmax(energy[i,:]) for i in 1:numcells] # Frame at which the delay is maximum

peaktobaseline = [energy[i,d]./energy[i,3] for (i,d) in enumerate(maxdelay)] # Ratio of energy in peak frame and baseline frame

SNR = reshape([mean(usta[cell,d,:].^2)./noise[cell] for cell in 1:numcells for (d,_) in enumerate(delays)], numcells, :)
SNR_peak = [SNR[i,d]./SNR[i,3] for (i,d) in enumerate(maxdelay)] # SNR of the peak frame relative to the noise frame

p = kde(log10.(SNR_peak))
histogram(log10.(SNR_peak), bins = 0:0.02:1.0, tickdir= :out)
plot!(p.x,p.density*15, linewidth=3,color=2,label="kde fit")

# Most of the cells have some SNR
# Next I want to color code the cells based on their locations =
cellLoc = zeros(Float64, cellNum,2)
for i in 1:cellNum
    cellLoc[i,:] = mean(cellRoi[i],dims=1);
end
Gadfly.plot(x = cellLoc[:,1], y = cellLoc[:,2], color = log10.(SNR_peak))

# Thresholding the cells based on their SNR_peak and seeing the 2D distribution
Gadfly.plot(x = cellLoc[:,1], y = cellLoc[:,2], color = SNR_peak.>1)

##-- Calculating the weighted STA: for improvement of the STA

weightedSTA = Array{Float64}(undef,cellNum,size(usta,3))
for i in 1:cellNum
    p = argmax(energy[i]) + 2
    if p==3
        w = [p p+1]
    elseif p==15
        w = [p-1 p]
    else
        w = [p-1 p p+1]
    end
    weights = energy[i,w]./norm(energy[i,w],2)

    weightedSTA[i,:] = weights * reshape(usta[i,w,:],length(weights),:)
end

## Plotting example STAs
@manipulate for cellid in 1:20
    STA = weightedSTA[cellid,:]
    r = [extrema(STA)...]
    t = "Weighted_STA_cellid_$(cellid)"
    p = plotsta(STA,sizepx=imagesize,sizedeg=stisize,ppd=ppd,index=xi,title=t,r=r)
end
