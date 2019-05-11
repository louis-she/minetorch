import 'babel-polyfill'
import Vue from 'vue'
import Vuex from 'vuex'
import ElementUI from 'element-ui'
import MintUI from 'mint-ui'
import i18n from './locale/i18n'

import 'element-ui/lib/theme-chalk/index.css'
import 'mint-ui/lib/style.css'

import App from './App'
import router from './router'
import storeMain from './store'
import { ajax } from 'utils/ajax'

Vue.use(Vuex)
Vue.use(ElementUI, {i18n: (key, value) => i18n.t(key, value)})
Vue.use(MintUI)

Vue.prototype.ajax = ajax
Vue.prototype.steps = ['dataset', 'dataflow', 'model', 'optimizer', 'loss']

const store = new Vuex.Store(storeMain)
const isDev = process.env.NODE_ENV === 'development'

/* eslint-disable no-new */
new Vue({
  el: '#app',
  store,
  router,
  i18n,
  render: h => h(App)
})

/*
  开发环境 开启 vue-devtools
 */
if (isDev) {
  Vue.config.devtools = true
}
