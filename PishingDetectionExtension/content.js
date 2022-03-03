/* 
Web traffic
SFH
link in tags
Url of anchors
domain registration length 
ssl final state
1 legit
-1 pishing
0 suspicious
*/


var website_url = window.location.href;
var SSL_final_state;
var having_Sub_domain;
var Url_of_anchors; 
var Prefix_Suffix;

function giveDomain(url){
    let domain = (new URL(url));
    domain = domain.hostname.replace('www.','');
    domain = domain.replace('.com.','');
    return domain;
}

/*Checking ssl state and alert if is true */
function SSLCheck(){
    isHTTPS = document.location.protocol == 'https:';
        if(isHTTPS){
        return SSL_final_state = 1;
    }else{
        alert("https is not enable");
    }
}

/*Checking subdomain amount */
function countSubdomains(){
    var host = window.location.host;
    for(let i = 0 ; i <=6; i ++){
        var count_subdomain = 0;
        var subdomain = host.split('.')[i];
        count_subdomain++;
        if (count_subdomain >= 5){
            return having_Sub_domain = -1;
        }else{
            return having_Sub_domain = 1; 
        }
    }
}

/*count url anchors */
function urlAnchors(){
    const listOfLinks = document.links;
    let url_anchors;
    let website_url_domain =  giveDomain(website_url);
    var count_TRUE = 0;

    for(let i =0 ; i < listOfLinks.length; i++){
        url_anchors =  giveDomain(listOfLinks[i]);
        if( url_anchors === website_url_domain){
                count_TRUE++;
        }
    }
    var percentage_true =(count_TRUE/listOfLinks.length)*100;
    console.log(percentage_true);
    if(percentage_true >= 69){
        return Url_of_anchors = 1; 
    }else{
        return Url_of_anchors = -1; 
    } 
}
SSLCheck();
countSubdomains();
urlAnchors();
if (window.location.href.indexOf("-") > -1) {
    Prefix_Suffix = -1;
}else{
    Prefix_Suffix = 1;
}
/*making the api request for prediction */
async function getPrediction(){
    let response = await fetch('http://127.0.0.1:5000/predict', { method: 'POST',headers: {
        "Content-Type": "application/json"
      }, body:JSON.stringify({ "SSLfinal_State":SSL_final_state, "URL_of_Anchor":Url_of_anchors, "having_Sub_Domain":having_Sub_domain, "Prefix_Suffix":Prefix_Suffix,}), });
    let data = await response.json();
    return data;
}
function alertPrediction(n){
    let answer = n.prediction[1];
    console.log(answer);
    if(answer === "1"){
        alert("is legit website, used carefully");
    }else if(answer === "-1"){
        alert("is a pishing website close the browser right now");
    }else{
        alert("suspichious website be careful");
    }
}
getPrediction().then(alertPrediction);
