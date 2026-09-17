const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const categories=['DRESSES','TOPS','CO-ORD SETS','JACKETS','ACCESSORIES','LOUNGEWEAR'];
const products=[['Linen Maxi Dress','₹2,499','DRESSES'],['Floral Wrap Dress','₹2,499','DRESSES'],['Satin Cowl Top','₹1,499','TOPS'],['Pleated Midi Skirt','₹1,899','SKIRTS'],['Linen Co-ord Set','₹2,799','CO-ORD SETS']];
const state={cart:[],wish:[],filter:'all'};
function renderCategories(){ $('#categories').innerHTML=categories.map(c=>`<div class="category-wrap"><button class="category" data-category="${c}"><img src="/ugc/assets/categories.svg" alt="${c}"></button><span>${c}</span></div>`).join('');}
function renderProducts(){ $('#products').innerHTML=products.filter(p=>state.filter==='all'||p[2]===state.filter).map((p,i)=>`<article class="product"><button class="heart" data-wish="${i}">${state.wish.includes(i)?'♥':'♡'}</button><div class="product-image"><img src="/ugc/assets/products.svg" alt="${p[0]}"></div><h3>${p[0]}</h3><p class="price">${p[1]}</p><div class="swatches"><i></i><i></i><i></i></div><button class="primary" data-add="${i}" style="padding:8px 10px;margin:0 4px">ADD TO BAG</button></article>`).join('')||'<p>No pieces in this collection yet.</p>';}
function openModal(title,body){$('#modal-content').innerHTML=`<h2>${title}</h2><p>${body}</p>`;$('#overlay').hidden=false}
function updateCart(){ $('#cart-count').textContent=state.cart.length; $('#cart-items').innerHTML=state.cart.length?state.cart.map(i=>`<p>${products[i][0]} — ${products[i][1]}</p>`).join(''):'<p>Your bag is empty.</p>';$('#cart-total').textContent='₹'+state.cart.reduce((s,i)=>s+Number(products[i][1].replace(/[^0-9]/g,'')),0).toLocaleString('en-IN');}
function showCart(){ $('#drawer').classList.add('open');updateCart();}
document.addEventListener('click',e=>{const t=e.target.closest('button,a');if(!t)return;
 if(t.dataset.add!==undefined){state.cart.push(+t.dataset.add);updateCart();showCart();}
 if(t.dataset.wish!==undefined){const i=+t.dataset.wish;state.wish=state.wish.includes(i)?state.wish.filter(x=>x!==i):[...state.wish,i];renderProducts();}
 if(t.dataset.category){state.filter=t.dataset.category;renderProducts();document.querySelector('#new').scrollIntoView();}
 if(t.dataset.filter){state.filter='all';renderProducts();document.querySelector('#new').scrollIntoView();}
 if(t.dataset.scroll){document.querySelector(t.dataset.scroll).scrollIntoView();}
 if(t.dataset.action==='search')openModal('Find your style','Search is ready — try “linen”, “dresses”, or “tops”. <input placeholder="Search products…">');
 if(t.dataset.action==='account')openModal('Welcome to MIRAE','Sign in to save your wishlist and track orders. <input placeholder="Email address" type="email"><button class="primary">CONTINUE</button>');
 if(t.dataset.action==='wishlist')openModal('Your Wishlist',state.wish.length?state.wish.map(i=>products[i][0]).join('<br>'):'Your wishlist is waiting for a little love ♡');
 if(t.dataset.action==='journal'||t.dataset.action==='read')openModal('From the Journal','Our editorial stories are coming soon. Subscribe for styling tips and new arrivals.');
 if(t.dataset.action==='contact')openModal('We’re here to help','Email us at hello@mirae.example and our team will get back to you.');
 if(t.dataset.action==='checkout')openModal('Checkout','Demo checkout: your bag is ready. Connect your payment provider to accept real orders.');
 if(t.dataset.action==='close')$('#overlay').hidden=true;
 if(t.dataset.action==='cart')showCart();
 if(t.dataset.action==='close-cart')$('#drawer').classList.remove('open');
});
$('#newsletter').addEventListener('submit',e=>{e.preventDefault();openModal('You’re on the list ♡','Thank you for subscribing to MIRAE updates.');e.target.reset();});
renderCategories();renderProducts();updateCart();