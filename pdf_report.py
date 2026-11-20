"""Generate downloadable PDF reports from saved model results (no retraining)."""
from io import BytesIO
from datetime import datetime, timezone
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Flowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

PALETTE={'Low':colors.HexColor('#238568'),'Moderate':colors.HexColor('#bf8b31'),'High':colors.HexColor('#b85f4b')}
NAVY=colors.HexColor('#193646')
def num(x):
    v=float(x)
    if not __import__('math').isfinite(v):raise ValueError('Invalid score')
    return v
def txt(v):return escape(str(v if v is not None else ''))
def report_parameter_labels():
    """Use the same CDOT labels exposed to the website by /schema."""
    from app import B, worksheet_score_labels
    labels = dict(B.get('item_names', {})) if isinstance(B, dict) else {}
    labels.update(worksheet_score_labels())
    return labels

STANDARD_PARAMETER_NAMES = {
    'rock_type':'Rock Type', 'slope_type':'Slope Type', 'region':'Engineering Region',
    'rater':'Inspector / Rater',
    'slope_height':'Slope Height (ft)', 'slope_angle':'Slope Angle (degrees)',
    'slope_length':'Slope Length (ft)', 'ditch_slope':'Ditch Slope (degrees)',
    'ditch_width':'Ditch Width (ft)', 'ditch_depth':'Ditch Depth (ft)',
    'sc28':'Rockfall Frequency Score (Slope)',
    'sc32':'Annual Precipitation Score (Climate)',
    'sc33':'Annual Freeze / Thaw Score (Climate)',
    'sc34':'Seepage / Water Score (Climate)',
    'sc35':'Slope Aspect Score (Climate)',
    'sc27':'Slope Height Score (Slope)',
    'sc29':'Slope Angle Score (Slope)',
    'sc31':'Ditch Catchment Score (Slope)',
    'sc51':'Sight Distance Score (Traffic)',
    'sc52':'Number of Accidents Score (Traffic)',
    'sc53':'Average Vehicle Risk Score (Traffic)',
}
def parameter_name(key, worksheet_labels):
    key=str(key)
    # Match the form's live /schema labels, especially for rock-specific scores.
    return worksheet_labels.get(key) or STANDARD_PARAMETER_NAMES.get(key) or key.replace('_',' ').title()

def fmt(v):return f'{num(v):,.1f}'
def classify(score,low,high):
    return 'Low' if score<=low else 'Moderate' if score<=high else 'High'
class Distribution(Flowable):
    def __init__(self,counts,width=470):super().__init__();self.counts=counts;self.width=width;self.height=92
    def draw(self):
        c=self.canv;n=max(1,sum(self.counts.values()))
        for i,label in enumerate(('Low','Moderate','High')):
            y=65-i*29;count=self.counts[label]
            c.setFillColor(NAVY);c.setFont('Helvetica',10);c.drawString(0,y,label)
            c.setFillColor(colors.HexColor('#edf1f2'));c.roundRect(82,y-2,250,12,5,fill=1,stroke=0)
            if count:
                c.setFillColor(PALETTE[label]);c.roundRect(82,y-2,250*count/n,12,5,fill=1,stroke=0)
            c.setFillColor(NAVY);c.drawString(345,y,f'{count} assessments')
class ScorePlot(Flowable):
    def __init__(self,rows):super().__init__();self.rows=rows[:16];self.width=470;self.height=24*len(self.rows)+20
    def draw(self):
        c=self.canv;maxscore=max([abs(x[1]) for x in self.rows]+[1])
        for i,(name,score,cls) in enumerate(self.rows):
            y=self.height-20-i*24
            c.setFont('Helvetica',8);c.setFillColor(NAVY);c.drawString(0,y,str(name)[:24])
            c.setFillColor(colors.HexColor('#edf1f2'));c.rect(150,y-2,230,10,fill=1,stroke=0)
            c.setFillColor(PALETTE[cls]);c.rect(150,y-2,230*max(0,score)/maxscore,10,fill=1,stroke=0)
            c.setFillColor(NAVY);c.drawRightString(465,y,f'{score:.1f}')
def create_pdf(records):
    if not isinstance(records,list) or not 1<=len(records)<=100:raise ValueError('Select 1 to 100 assessments')
    clean=[]
    for r in records:
        if not isinstance(r,dict) or not isinstance(r.get('result'),dict):raise ValueError('Invalid assessment')
        result=r['result'];score=num(result['predicted_total_score']);low=num(result['low_cutoff']);high=num(result['high_cutoff'])
        if low>high:raise ValueError('Invalid thresholds')
        cls=classify(score,low,high)
        clean.append({'name':str(r.get('name','Untitled slope'))[:120],'time':str(r.get('time',''))[:40],'inputs':r.get('inputs') if isinstance(r.get('inputs'),dict) else {},'score':score,'low':low,'high':high,'class':cls})
    buf=BytesIO();doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=45)
    st=getSampleStyleSheet()
    st.add(ParagraphStyle(name='MainTitleX',parent=st['Title'],fontSize=19,leading=23,textColor=NAVY,spaceAfter=8))
    st.add(ParagraphStyle(name='HeadingX',parent=st['Heading2'],fontSize=12,leading=15,textColor=NAVY,spaceBefore=17,spaceAfter=9))
    st.add(ParagraphStyle(name='BodyX',parent=st['BodyText'],fontSize=9,leading=14,spaceAfter=8))
    st.add(ParagraphStyle(name='SmallX',parent=st['BodyText'],fontSize=7.5,leading=11))
    P=lambda s,sty='BodyX':Paragraph(s,st[sty])
    story=[P('HIGHWAY ROCKFALL<br/>HAZARD ASSESSMENT REPORT','MainTitleX'),P('Model: v4 k=9 Gradient Boosting + Ridge ensemble | Generated: '+datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC'),'SmallX'),Spacer(1,12)]
    counts={k:sum(r['class']==k for r in clean) for k in ('Low','Moderate','High')}
    story += [P(f'<b>Total assessments:</b> {len(clean)} &nbsp;&nbsp; <b>Low:</b> {counts["Low"]} &nbsp;&nbsp; <b>Moderate:</b> {counts["Moderate"]} &nbsp;&nbsp; <b>High:</b> {counts["High"]}'),P('1. Hazard Class Distribution','HeadingX'),Distribution(counts),P('The chart displays assessment counts only.','SmallX'),P('2. Predicted Score Comparison','HeadingX'),ScorePlot([(r['name'],r['score'],r['class']) for r in clean]),P('Each bar represents a predicted total hazard score in score units. The chart displays up to 16 assessments.','SmallX'),P('3. How the Hazard Class Is Calculated','HeadingX'),P('The model uses the submitted slope geometry, environmental, traffic and geological assessment values to estimate a total hazard score. The result is compared with the Low and High cutoffs returned by the model for that assessment.'),P('<b>Classification ranges (exact rules used by the backend):</b>'),P('LOW: score &le; Low cutoff<br/>MODERATE: Low cutoff &lt; score &le; High cutoff<br/>HIGH: score &gt; High cutoff.')]
    for i,r in enumerate(clean,1):
        s,l,h=r['score'],r['low'],r['high'];cls=r['class']
        story.append(P(f'Assessment {i}: {txt(r["name"])}','HeadingX'))
        story.append(P('MODEL RESULT — SCORE-BASED HAZARD CLASSIFICATION','SmallX'))
        hero=Table([
            [P('<font color="#b85f4b"><b>'+cls.upper()+'</b></font>','MainTitleX'),P('<b>'+fmt(s)+'</b>','MainTitleX')],
            [P('Predicted hazard class','SmallX'),P('Predicted total score','SmallX')]
        ],colWidths=[245,245],hAlign='LEFT')
        hero.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f7f9f9')),('LEFTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,0),13),('BOTTOMPADDING',(0,1),(-1,1),12),('LINEBELOW',(0,-1),(-1,-1),0.6,colors.HexColor('#e3e8e8'))]))
        story.append(hero)
        ranges=[('Low hazard','Score &le; '+fmt(l)),('Moderate hazard',fmt(l)+' &lt; Score &le; '+fmt(h)),('High hazard','Score &gt; '+fmt(h))]
        rr=Table([[P('<b>'+label+'</b>','BodyX'),P(value,'BodyX')] for label,value in ranges],colWidths=[245,245],hAlign='LEFT')
        rr.setStyle(TableStyle([('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,colors.HexColor('#f8f9f9')]),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),0.4,colors.HexColor('#e1e7e8'))]))
        story.extend([Spacer(1,10),rr,Spacer(1,10)])
        if cls=='High':
            difference=s-h;label='Above High cutoff';sentence=f'The predicted score exceeds the High-hazard boundary by {fmt(difference)} score points. Therefore, the model assigns the High Hazard class.'
        elif cls=='Low':
            difference=l-s;label='Below Low cutoff';sentence=f'The predicted score is {fmt(difference)} score points below the Low-hazard boundary. Therefore, the model assigns the Low Hazard class.'
        else:
            difference=s-l;label='Above Low cutoff';sentence=f'The predicted score is {fmt(difference)} points above the Low cutoff and {fmt(h-s)} points below the High cutoff. Therefore, the model assigns the Moderate Hazard class.'
        row=Table([[P('<b>'+label+'</b>'),P('<font color="#b85f4b"><b>+'+fmt(difference)+' points</b></font>')]],colWidths=[245,245],hAlign='LEFT')
        row.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#fff4ed')),('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12)]))
        story.extend([row,Spacer(1,9),P(sentence),P('The difference is a distance from the classification cutoff, not a measured increase in rockfall probability.','SmallX')])
        if r['time']:story.append(P('Recorded: '+txt(r['time']),'SmallX'))
        story.append(P('All Submitted Input Parameters','HeadingX'))
        labels=report_parameter_labels()
        entries=list(r['inputs'].items())
        if entries:
            rows=[[P('<b>Input parameter</b>','SmallX'),P('<b>Submitted value</b>','SmallX')]]
            rows.extend([[P(txt(parameter_name(k, labels)),'SmallX'),P(txt(v),'SmallX')] for k,v in entries])
            t=Table(rows,colWidths=[245,245],repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0ef')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9f9')]),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('VALIGN',(0,0),(-1,-1),'TOP')]))
            story.append(t)
        else:story.append(P('No submitted input values were saved with this assessment.'))
    story += [P('4. Interpretation and Limitations','HeadingX'),P('A High classification indicates a predicted total score above the applicable High cutoff. It does not mean a rockfall will definitely occur. A hazard score is NOT a percentage, calibrated probability, or event forecast. Assessments from different slopes cannot establish a change in hazard over time at a single site.'),P('This is a model-based screening report, not a certified geotechnical safety assessment. Decisions affecting public safety require field verification and qualified professional review.')]
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey);canvas.drawString(43,26,'Highway Rockfall Hazard Classification | v4 k=9');canvas.drawRightString(A4[0]-43,26,f'Page {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);return buf.getvalue()
