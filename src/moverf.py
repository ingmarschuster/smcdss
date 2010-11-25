#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
USAGE: moverf [OPTIONS]

    -h, --help
                                Zeigt diese Nachricht an.
                                
    -f, --file
                                Dateiname mit vollständigem Pfad.
                                
    -n, --number
                                Anzahl der zufälligen Grafiken.
'''

from math import log
from random import normalvariate, random
import getopt, os, sys, time

# top line
y_top = 34

# bottom line
y_bottom = 223.0

def header():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="710"
   height="240"
   id="svg2"
   version="1.1"
   inkscape:version="0.46"
   sodipodi:docname="Umsatzgrafik.svg"
   sodipodi:version="0.32"
   inkscape:output_extension="org.inkscape.output.svg.inkscape">
  <defs
     id="defs4">
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 120 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="710 : 120 : 1"
       inkscape:persp3d-origin="355 : 80 : 1"
       id="perspective139" />
    <linearGradient
       gradientTransform="matrix(1,0,0,-1,-66,62.5)"
       y2="-133.3584"
       x2="259"
       y1="-133.3584"
       x1="124"
       gradientUnits="userSpaceOnUse"
       id="SVGID_6_">
      <stop
         id="stop3144"
         style="stop-color:#ff9f02;stop-opacity:1;"
         offset="0" />
      <stop
         id="stop3146"
         style="stop-color:#FFBD02"
         offset="1" />
    </linearGradient>
    <linearGradient
       gradientTransform="matrix(1,0,0,-1,-66,62.5)"
       y2="-79.250504"
       x2="259"
       y1="-79.250504"
       x1="124"
       gradientUnits="userSpaceOnUse"
       id="SVGID_7_">
      <stop
         id="stop3151"
         style="stop-color:#FFE600"
         offset="0" />
      <stop
         id="stop3153"
         style="stop-color:#FFEF57"
         offset="1" />
    </linearGradient>
    <filter
       inkscape:collect="always"
       id="filter3575">
      <feGaussianBlur
         inkscape:collect="always"
         stdDeviation="5.7916706"
         id="feGaussianBlur3577" />
    </filter>
    <filter
       inkscape:collect="always"
       id="filter3587">
      <feGaussianBlur
         inkscape:collect="always"
         stdDeviation="2.9124999"
         id="feGaussianBlur3589" />
    </filter>
    <filter
       inkscape:collect="always"
       id="filter3635">
      <feGaussianBlur
         inkscape:collect="always"
         stdDeviation="3.7832201"
         id="feGaussianBlur3637" />
    </filter>
    <filter
       inkscape:collect="always"
       id="filter3639"
       x="-0.042680412"
       width="1.0853608"
       y="-0.10097561"
       height="1.2019512">
      <feGaussianBlur
         inkscape:collect="always"
         stdDeviation="5.175"
         id="feGaussianBlur3641" />
    </filter>
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_6_"
       id="linearGradient3796"
       gradientUnits="userSpaceOnUse"
       x1="370.84409"
       y1="420.1604"
       x2="658.84412"
       y2="420.1604"
       gradientTransform="translate(-1.8440933,-257.1604)" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_6_"
       id="linearGradient3802"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,-0.8842549,-65.999998,81.07711)"
       x1="124"
       y1="-133.3584"
       x2="259"
       y2="-133.3584" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_6_"
       id="linearGradient3806"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,0.7913544,-1.8440933,-157.24802)"
       x1="207.84409"
       y1="419.84729"
       x2="342.84409"
       y2="419.84729" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_7_"
       id="linearGradient3810"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,1.613645,-1.8440933,-452.60804)"
       x1="207.84409"
       y1="338.84732"
       x2="342.84409"
       y2="338.84732" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_7_"
       id="linearGradient3815"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,-0.8971794,-65.999998,79.89807)"
       x1="124"
       y1="-79.250504"
       x2="259"
       y2="-79.250504" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_7_"
       id="linearGradient3821"
       gradientUnits="userSpaceOnUse"
       x1="369.34409"
       y1="395.49356"
       x2="660.34412"
       y2="395.49356"
       gradientTransform="translate(-1.8440933,-257.1604)" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_6_"
       id="linearGradient3841"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,-1.0496851,-65.999998,65.64834)"
       x1="483.89651"
       y1="52.538101"
       x2="503.89651"
       y2="52.538101" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_7_"
       id="linearGradient3871"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,-1,-65.999998,62.8424)"
       x1="71"
       y1="54.5"
       x2="77"
       y2="54.5" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_6_"
       id="linearGradient3874"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(1,0,0,-1,-65.999998,62.8424)"
       x1="134.625"
       y1="54.5"
       x2="140.625"
       y2="54.5" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_7_"
       id="linearGradient3300"
       x1="369"
       y1="9.9624996"
       x2="387.76999"
       y2="9.9624996"
       gradientUnits="userSpaceOnUse" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#SVGID_7_"
       id="linearGradient2521"
       gradientUnits="userSpaceOnUse"
       x1="369"
       y1="9.9624996"
       x2="387.76999"
       y2="9.9624996"
       gradientTransform="matrix(1,0,0,0.1438228,1.6772461e-6,7.7727596)" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="1.3043168"
     inkscape:cx="370.05987"
     inkscape:cy="222.52575"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="true"
     showguides="true"
     inkscape:guide-bbox="true"
     fit-margin-top="0"
     fit-margin-left="0"
     fit-margin-right="0"
     fit-margin-bottom="0"
     inkscape:snap-grids="true"
     inkscape:window-width="1280"
     inkscape:window-height="948"
     inkscape:window-x="0"
     inkscape:window-y="0"
     inkscape:window-maximized="1"
     gridtolerance="50"
     guidetolerance="50">
    <inkscape:grid
       type="xygrid"
       id="grid5949"
       empspacing="5"
       visible="true"
       enabled="true"
       snapvisiblegridlinesonly="true"
       spacingx="24px"
       spacingy="24px" />
  </sodipodi:namedview>
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Ebene 1"
     inkscape:groupmode="layer"
     id="layer1">
    <rect
       style="fill:#e9e9e5"
       x="0"
       y="0"
       width="710"
       height="240"
       id="rect2993" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="34.342407"
       x2="346.52899"
       y2="34.342407"
       id="line2995" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="61.342377"
       x2="346.52899"
       y2="61.342377"
       id="line2997" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="88.342377"
       x2="346.52899"
       y2="88.342377"
       id="line2999" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="115.34238"
       x2="346.52899"
       y2="115.34238"
       id="line3001" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="142.34238"
       x2="346.52899"
       y2="142.34238"
       id="line3003" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="169.34238"
       x2="346.52899"
       y2="169.34238"
       id="line3005" />
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.0000019"
       y1="196.34238"
       x2="346.52899"
       y2="196.34238"
       id="line3007" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="34.342407"
       x2="705.86298"
       y2="34.342407"
       id="line3009" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="61.636383"
       x2="705.86298"
       y2="61.636383"
       id="line3011" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="88.342377"
       x2="705.86298"
       y2="88.342377"
       id="line3013" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="115.34238"
       x2="705.86298"
       y2="115.34238"
       id="line3015" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="142.34238"
       x2="705.86298"
       y2="142.34238"
       id="line3017" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="169.34238"
       x2="705.86298"
       y2="169.34238"
       id="line3019" />
    <line
       style="fill:none;stroke:#ffffff;stroke-dasharray:1, 3, 1, 3, 1, 3"
       x1="364.33401"
       y1="196.34238"
       x2="705.86298"
       y2="196.34238"
       id="line3021" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98623502;stroke-dasharray:0.986235, 2.95870499, 0.986235, 2.95870499, 0.986235, 2.95870499"
       x1="393"
       y1="4.9417114"
       x2="393"
       y2="223.96906"
       id="line3027" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98623067;stroke-dasharray:0.98623067, 2.95869202, 0.98623067, 2.95869202, 0.98623067, 2.95869202"
       x1="369"
       y1="4.3287048"
       x2="369"
       y2="223.35605"
       id="line3029" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="416.9931"
       y1="4.681488"
       x2="416.9931"
       y2="223.69528"
       id="line3031" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="440.99451"
       y1="4.681488"
       x2="440.99451"
       y2="223.69528"
       id="line3033" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="464.99591"
       y1="4.681488"
       x2="464.99591"
       y2="223.69528"
       id="line3035" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="488.99731"
       y1="4.681488"
       x2="488.99731"
       y2="223.69528"
       id="line3037" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="512.99872"
       y1="4.681488"
       x2="512.99872"
       y2="223.69528"
       id="line3039" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="537.00012"
       y1="4.681488"
       x2="537.00012"
       y2="223.69528"
       id="line3041" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="561.00153"
       y1="4.681488"
       x2="561.00153"
       y2="223.69528"
       id="line3043" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="585.00293"
       y1="4.681488"
       x2="585.00293"
       y2="223.69528"
       id="line3045" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="609.00427"
       y1="4.681488"
       x2="609.00427"
       y2="223.69528"
       id="line3047" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="657"
       y1="4.681488"
       x2="657"
       y2="223.69528"
       id="line3049" />
    <line
       style="fill:none;stroke:#ffffff;stroke-width:0.98622662;stroke-dasharray:0.98622663, 2.9586799, 0.98622663, 2.9586799, 0.98622663, 2.9586799"
       x1="633.00568"
       y1="4.681488"
       x2="633.00568"
       y2="223.69528"
       id="line3051" />
    <text
       id="text3099"
       font-size="8"
       style="font-size:8px;font-family:Tahoma"
       x="399.19424"
       y="11.573905">YTD</text>
    <text
       id="text3103"
       font-size="8"
       style="font-size:8px;font-family:Tahoma"
       x="450.16019"
       y="11.237754">Vorjahr</text>
    <text
       id="text3107"
       font-size="8"
       style="font-size:8px;font-family:Tahoma"
       x="15.364702"
       y="11.342407">International</text>
    <text
       id="text3111"
       font-size="8"
       style="font-size:8px;font-family:Tahoma"
       x="77.977501"
       y="11.342407">National</text>
    <rect
       id="rect3120"
       height="6"
       width="6"
       y="5.3424072"
       x="68.625"
       style="fill:url(#linearGradient3874);fill-opacity:1" />
    <rect
       style="fill:url(#linearGradient3871);fill-opacity:1"
       x="5.0000019"
       y="5.3424072"
       width="6"
       height="6"
       id="rect3133" />
    <rect
       style="fill:none"
       x="55.527"
       y="189.61938"
       width="140.741"
       height="28.149"
       id="rect3179" />
    <rect
       style="fill:none"
       x="209.82498"
       y="154.43637"
       width="126.667"
       height="28.146999"
       id="rect3183" />
    <rect
       style="fill:none"
       x="209.82498"
       y="76.659393"
       width="126.667"
       height="28.149"
       id="rect3187" />
    <rect
       style="fill:none"
       x="61.684998"
       y="138.24237"
       width="126.667"
       height="28.146999"
       id="rect3191" />
    <rect
       style="fill:none"
       x="4.736002"
       y="172.5004"
       width="84.445"
       height="28.148001"
       id="rect3199" />
    <rect
       style="fill:none"
       x="4.736002"
       y="199.5004"
       width="84.445"
       height="28.148001"
       id="rect3203" />
    <rect
       style="fill:none"
       x="4.736002"
       y="145.5004"
       width="84.445"
       height="28.148001"
       id="rect3207" />
    <rect
       style="fill:none"
       x="4.736002"
       y="118.5004"
       width="84.445"
       height="28.148001"
       id="rect3211" />
    <rect
       style="fill:none"
       x="4.736002"
       y="91.49939"
       width="84.445"
       height="28.149"
       id="rect3215" />
    <rect
       style="fill:none"
       x="4.736002"
       y="64.500397"
       width="84.445"
       height="28.148001"
       id="rect3219" />
    <rect
       style="fill:none"
       x="4.736002"
       y="37.500397"
       width="84.445"
       height="28.148001"
       id="rect3223" />
    <rect
       id="rect3229"
       height="28.148001"
       width="50.778"
       y="172.5004"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3233"
       height="28.148001"
       width="50.778"
       y="199.5004"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3237"
       height="28.148001"
       width="50.778"
       y="145.5004"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3241"
       height="28.148001"
       width="50.778"
       y="118.5004"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3245"
       height="28.149"
       width="50.778"
       y="91.49939"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3249"
       height="28.148001"
       width="50.778"
       y="64.500397"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3253"
       height="28.148001"
       width="50.778"
       y="37.500397"
       x="654.11798"
       style="fill:none" />
    <rect
       id="rect3270"
       height="2.9999998"
       width="18.77"
       y="9"
       x="369"
       style="opacity:0.75;fill:url(#linearGradient2521);fill-opacity:1" />
    <line
       id="line3278"
       y2="8.5"
       x2="388.591"
       y1="8.5"
       x1="369"
       style="fill:none;stroke:#000000;stroke-width:3;stroke-dasharray:3, 5, 3, 5, 3, 5" />
    <rect
       id="rect3293"
       height="3"
       width="20"
       y="9"
       x="417.896"
       style="fill:url(#linearGradient3841);fill-opacity:1" />
    <line
       id="line3301"
       y2="8.5"
       x2="437.896"
       y1="8.5"
       x1="417.896"
       style="fill:none;stroke:#000000;stroke-width:3" />
    <text
       style="font-size:14.07419968px;font-family:Tahoma-Bold"
       font-size="14.0742"
       id="text3307"
       x="98.433098"
       y="235.53064">Vorjahr</text>
    <text
       style="font-size:14.07419968px;font-family:Tahoma-Bold"
       font-size="14.0742"
       id="text3311"
       x="259.8027"
       y="236.48096">YTD</text>
    <g
       id="monate">
      <text
         id="text3325"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="372.91406"
         y="233.33594">Jan.</text>
      <text
         id="text3331"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="396.50787"
         y="233.33594">Feb.</text>
      <text
         id="text3337"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="420.60938"
         y="233.33594">Mrz.</text>
      <text
         id="text3343"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="444.85938"
         y="233.33594">Apr.</text>
      <text
         id="text3349"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="470.02536"
         y="233.33594">Mai</text>
      <text
         id="text3355"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="493.08008"
         y="233.33594">Juni</text>
      <text
         id="text3361"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="518.39648"
         y="233.33594">Juli</text>
      <text
         id="text3367"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="540.07043"
         y="233.33594">Aug.</text>
      <text
         id="text3373"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="564.23828"
         y="233.33594">Sept</text>
      <text
         id="text3379"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="588.75012"
         y="233.33594">Okt.</text>
      <text
         id="text3385"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="612.07623"
         y="233.33594">Nov.</text>
      <text
         id="text3391"
         font-size="8"
         style="font-size:8px;fill:#333333;font-family:Tahoma"
         x="636.31451"
         y="233.33594">Dez.</text>
    </g>'''

def footer():
    return '''
    <line
       style="fill:none;stroke:#ffffff"
       x1="5.2640018"
       y1="223.3396"
       x2="705"
       y2="223.3396"
       id="line3313"/>
  </g>
</svg>
    '''

def scale_max(v_max):
    return scale_shift(v_max) * 7.0

def scale_shift(v_max):
    scale = .28 * 10 ** int(log(v_max) / log(10))
    return (round(int(v_max) / scale + 0.5)) * scale / 7.0

def create_scale(v_max):

    y_shift = -27
    s_shift = scale_shift(v_max)
    scale = [i * s_shift for i in range(1, 8)]
    xml = ''

    for side in [('links', 4.5), ('rechts', 662)]:
        xml += '''
    <g
       id="skala_%s"
       inkscape:label="#skala_%s">'''\
        % (side[0], side[0])
        y = 204
        for i, value in enumerate(scale):
            xml += '''
      <text
         y="%.1f"
         x="%.1f"
         id="text_skala_%s_%i"
         font-size="8"
         style="font-size:8px;fill:#4d4d4d;font-family:Tahoma">%.3f €</text>'''\
            % (y, side[1], side[0], i + 1, value)
            y += y_shift
        xml += '\n    </g>'
    return xml

def create_boxes(int_vor, nat_vor, int_ytd, nat_ytd):

    # value at top line
    v_max = max(int_vor + nat_vor, int_ytd + nat_ytd)
    scale = (y_bottom - y_top) / scale_max(v_max)

    xml = '''
    <rect
       style="opacity:0.35;fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:3;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;filter:url(#filter3587)"
       width="135"
       height="%.1f"
       x="58"
       y="%.1f"
       id="box_vorjahr_schatten"
       inkscape:label="#rect3579" />
    <rect
       id = "box_nat_vorjahr"
       height = "%.1f"
       width = "135"
       y = "%.1f"
       x = "58"
       style = "opacity:0.95;fill:url(#linearGradient3802);fill-opacity:1;stroke:none"
       inkscape:label = "#box_nat_vorjahr" />
    <rect
       id = "box_int_vorjahr"
       height = "%.1f"
       width = "135"
       x = "58"
       y = "%.1f"
       style = "opacity:0.95;fill:url(#linearGradient3815);stroke:none"
       inkscape:label = "#box_int_vorjahr" />'''\
    % ((nat_vor + int_vor) * scale, y_bottom - (nat_vor + int_vor) * scale,
       nat_vor * scale, y_bottom - nat_vor * scale,
       int_vor * scale, y_bottom - (nat_vor + int_vor) * scale)

    xml += '''
    <rect
       style = "opacity:0.35;fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:3;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;filter:url(#filter3635)"
       width = "135"
       height = "%.1f"
       x = "206"
       y = "%.1f"
       id = "box_ytd_schatten"
       inkscape:label = "#box_ytd_schatten" />
    <rect
       id = "box_nat_ytd"
       height = "%.1f"
       width = "135"
       y = "%.1f"
       x = "206"
       style = "opacity:0.95;fill:url(#linearGradient3806);fill-opacity:1;stroke:none"
       inkscape:label = "#box_nat_ytd" />
    <rect
       id = "box_int_ytd"
       inkscape:label = "#box_int_ytd"
       height = "%.1f"
       width = "135"
       y = "%.1f"
       x = "206"
       style = "opacity:0.95;fill:url(#linearGradient3810);fill-opacity:1.0;stroke:none" />'''\
    % ((nat_ytd + int_ytd) * scale, y_bottom - (nat_ytd + int_ytd) * scale,
       nat_ytd * scale, y_bottom - nat_ytd * scale,
       int_ytd * scale, y_bottom - (nat_ytd + int_ytd) * scale)

    for caption in \
    ((16, 16, 'ges_vor', 125.0, y_bottom - (nat_vor + int_vor) * scale - 7.0, 'ges_vor', int_vor + nat_vor),
     (12, 12, 'nat_vor', 125.0, y_bottom - (nat_vor / 2.0) * scale + 4.0, 'nat_vor', nat_vor),
     (12, 12, 'int_vor', 125.0, y_bottom - (nat_vor + int_vor / 2.0) * scale + 4.0, 'int_vor', int_vor),
     (16, 16, 'ges_ytd', 273.0, y_bottom - (nat_ytd + int_ytd) * scale - 7.0, 'ges_ytd', int_ytd + nat_ytd),
     (12, 12, 'nat_ytd', 273.0, y_bottom - (nat_ytd / 2.0) * scale + 4.0, 'nat_ytd', nat_ytd),
     (12, 12, 'int_ytd', 273.0, y_bottom - (nat_ytd + int_ytd / 2.0) * scale + 4.0, 'int_ytd', int_ytd)):
        xml += '''
    <text
       style="font-size:%ipx;text-align:center;text-anchor:middle;fill:#1a1a1a;font-family:Tahoma-Bold"
       font-size="%i"
       id="text_%s"
       x="%.1f"
       y="%.1f"
       inkscape:label="#text_%s" >%.3f €</text>'''\
       % caption

    return xml

def path_header(name):
    return '''    <path
       id="pfad_%s"
       inkscape:label="#pfad_%s"  
       inkscape:connector-curvature="0"
       sodipodi:nodetypes="cccccccccccccc"''' % (name, name)

def cum_sum(list, scale):
    n = len(list)
    for i in xrange(1, n): list[i] += list[i - 1]
    for i in xrange(0, n): list[i] = y_bottom - list[i] * scale
    return list

def create_path(mon_vor, mon_ytd):

    scale = (y_bottom - y_top) / scale_max(max(sum(mon_vor), sum(mon_ytd)))

    x = range(393, 658, 24)

    y = dict(
    ytd=cum_sum(mon_ytd, scale),
    vor=cum_sum(mon_vor, scale)
    )

    style = dict(
    ytd_schatten="opacity:0.5;fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;filter:url(#filter3575)",
    ytd_fuellung="fill:url(#linearGradient3821);fill-opacity:1;stroke:none;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1",
    ytd_linie="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:12, 6;stroke-dashoffset:0;stroke-opacity:1",
    vor_schatten="opacity:0.35;fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;filter:url(#filter3639)",
    vor_fuellung="fill:url(#linearGradient3796);fill-opacity:1;stroke:none;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1",
    vor_linie="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
    )

    xml = ''
    for name in [('ytd', 'schatten'), ('ytd', 'fuellung'), ('vor', 'schatten'), ('vor', 'fuellung'), ('vor', 'linie'), ('ytd', 'linie')]:
        xml += path_header(name[0] + '_' + name[1]) + '\n'
        xml += '       style="' + style[name[0] + '_' + name[1]] + '"\n'
        xml += '       d="M 369,223 L ' + ' L '.join([('%i,%.3f' % k) for k in zip(x, y[name[0]])])
        if name[1] == 'linie':xml += '" />\n'
        else: xml += 'L %i,223" />\n' % x[len(y[name[0]]) - 1]

    return xml

def create_xml(int_vor, nat_vor, int_ytd, nat_ytd, mon_vor, mon_ytd):
    xml = header()
    xml += create_scale(max(int_vor + nat_vor, int_ytd + nat_ytd))
    xml += create_boxes(int_vor, nat_vor, int_ytd, nat_ytd)
    xml += create_path(mon_vor, mon_ytd)
    xml += footer()
    return xml

def from_random_data():
    scale = abs(normalvariate(35.0, 10.0))
    mon_vor = [abs(normalvariate(1.5 * scale, scale)) for i in xrange(12)]
    s_vor = sum(mon_vor)
    r_vor = random()
    m = 5 + int(random()*8)
    mon_ytd = [abs(normalvariate(2.0 * scale, scale)) for i in xrange(m)]
    s_ytd = sum(mon_ytd)
    r_ytd = random()
    return create_xml(int_vor=r_vor * s_vor,
                       nat_vor=(1 - r_vor) * s_vor,
                       int_ytd=r_ytd * s_ytd,
                       nat_ytd=(1 - r_ytd) * s_ytd,
                       mon_vor=mon_vor,
                       mon_ytd=mon_ytd)

def main():

    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:n:", ['help', 'file=', 'number='])
    except getopt.error, msg:
        print msg
        sys.exit(2)

    n = 1
    f = 'out'

    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-f", "--file"):
            f = a
            if f.find('.') > -1: f = f[:f.find('.')]
        if o in ("-n", "--number"):
            n = int(a)

    t = time.time()
    for i in range(1, n + 1):
        print '%s%04i.png' % (f, i)
        file = open('%s%04i.svg' % (f, i), 'w')
        file.write(from_random_data())
        file.close()
        os.system('inkscape %s%04i.svg -e %s%04i.png -d 90 > /dev/null' % (f, i, f, i))
        os.system('rm %s%04i.svg' % (f, i))
    print '\n%i Grafik(en) erzeugt in %.2f Sekunden.' % (n, time.time() - t)

if __name__ == "__main__":
    main()
