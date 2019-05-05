
export default {
  namespaced: true,
  state: {
    /* 面包屑导航 */
    titlePath: [],
    /* 菜单路劲 router */
    routerPath: [],
    /* 菜单栏导航 */
    menu: [
      {
        icon: 'el-icon-menu',
        index: 'list',
        title: '项目管理'
      }
    ]
  },
  mutations: {
    /*
      routerPath 更新方法
     */
    routerPathPop (state) {
      state.routerPath.pop()
    },
    routerPathPush (state, val) {
      state.routerPath.push(val)
    },
    routerPathUpdate (state, data = []) {
      state.routerPath = data
    },
    routerPathClear (state) {
      state.routerPath = []
    },
    /*
      titlePath 更新方法
     */
    titlePathPop (state) {
      state.titlePath.pop()
    },
    titlePathPush (state, val) {
      state.titlePath.push(val)
    },
    titlePathUpdate (state, data = []) {
      state.titlePath = data
    },
    titlePathClear (state) {
      state.titlePath = []
    }
  },
  actions: {
  }
}
